#!/usr/bin/env python3
"""Convert a sphere-collision URDF to Horizon's stable sphere-model JSON."""
import argparse
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET


def parse_model(path, canonical_path=None):
    root = ET.parse(path).getroot()
    if root.tag != "robot" or not root.attrib.get("name"):
        raise ValueError("expected a named <robot> URDF")
    if canonical_path is None and root.attrib.get("path"):
        candidate = pathlib.Path(path).resolve().parent / root.attrib["path"]
        if candidate.is_file():
            canonical_path = candidate
    canonical_links = None
    if canonical_path is not None:
        canonical = ET.parse(canonical_path).getroot()
        canonical_links = {link.attrib["name"] for link in canonical.findall("link")}
    links, total, unsupported, omitted = [], 0, [], []
    for link in root.findall("link"):
        if canonical_links is not None and link.attrib.get("name") not in canonical_links:
            if link.findall("collision"):
                omitted.append(link.attrib.get("name", "<unnamed>"))
            continue
        spheres = []
        for i, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            sphere = geometry.find("sphere") if geometry is not None else None
            if sphere is None:
                unsupported.append(f"{link.attrib.get('name', '?')}[{i}]")
                continue
            radius = float(sphere.attrib["radius"])
            origin = collision.find("origin")
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
            if len(xyz) != 3 or len(rpy) != 3 or not all(map(math.isfinite, xyz + rpy)):
                raise ValueError(f"invalid origin on {link.attrib.get('name')} collision {i}")
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"invalid sphere radius on {link.attrib.get('name')} collision {i}")
            spheres.append({"center": xyz, "radius": radius})
        if spheres:
            links.append({"name": link.attrib["name"], "spheres": spheres})
            total += len(spheres)
    if not total:
        raise ValueError("no sphere collision geometry found")
    link_names = {link["name"] for link in links}
    missing = sorted(link_names - canonical_links) if canonical_links is not None else []
    return {"format": "horizon.sphere_model.v1", "robot": root.attrib["name"],
            "source": pathlib.Path(path).name, "sphere_count": total,
            "canonical_urdf": pathlib.Path(canonical_path).name if canonical_path else None,
            "links": links, "unsupported_collisions": unsupported,
            "links_missing_from_canonical": missing,
            "omitted_optional_links": sorted(omitted)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdf", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    ap.add_argument("--canonical-urdf", type=pathlib.Path,
                    help="robot URDF whose link names validate the sphere model")
    args = ap.parse_args()
    try:
        model = parse_model(args.urdf, args.canonical_urdf)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model, indent=2) + "\n")
    except (OSError, ET.ParseError, ValueError, KeyError) as exc:
        print(f"sphere model generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"generated {model['sphere_count']} spheres across {len(model['links'])} links: {args.output}")
    if model["unsupported_collisions"]:
        print(f"skipped unsupported collision elements: {len(model['unsupported_collisions'])}")
    if model["links_missing_from_canonical"]:
        print("sphere links absent from canonical URDF: " + ", ".join(model["links_missing_from_canonical"]))
    if model["omitted_optional_links"]:
        print("omitted collision links not present in canonical URDF: " + ", ".join(model["omitted_optional_links"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
