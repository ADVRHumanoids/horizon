#!/usr/bin/env python3
"""Experimental native cuRoboV2 FK + quaternion free-base collision adapter.

This probe does not register a production backend. FP32 native FK is validated
against Pinocchio; collision arithmetic and Gauss-Newton accumulation use FP64.
CUDA-graph timings exclude setup, upload/download and CPU reference evaluation.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import torch


class Request(ct.Structure):
    _fields_ = [(n, ct.c_uint64) for n in ("q", "positions", "quaternions", "jacobians", "stream")] + [
        ("batch_size", ct.c_int), ("horizon", ct.c_int)]


def rotation(q):
    """wxyz quaternion to rotation; leading dimensions are arbitrary."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def skew(p):
    x, y, z = p.unbind(-1); zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*p.shape[:-1], 3, 3)


class Native:
    def __init__(self, library, artifact, nodes):
        self.library = ct.CDLL(str(library))
        lib = self.library
        lib.curobo_robot_last_error.restype = ct.c_char_p
        lib.curobo_robot_create.argtypes = [ct.c_char_p, ct.c_int, ct.POINTER(ct.c_void_p)]
        lib.curobo_robot_compute_kinematics.argtypes = [ct.c_void_p, ct.POINTER(Request)]
        lib.curobo_robot_destroy.argtypes = [ct.c_void_p]
        self.handle = ct.c_void_p()
        self.check(lib.curobo_robot_create(str(artifact).encode(), 0, ct.byref(self.handle)))
        self.manifest = json.loads((artifact / "robot_kinematics_manifest.json").read_text())
        dof = self.manifest["num_dof"]; frames = self.manifest["num_tool_frames"]
        self.positions = torch.empty((nodes, frames, 3), device="cuda", dtype=torch.float32)
        self.quaternions = torch.empty((nodes, frames, 4), device="cuda", dtype=torch.float32)
        self.jacobians = torch.empty((nodes, frames, 6, dof), device="cuda", dtype=torch.float32)

    def check(self, status):
        if status:
            raise RuntimeError(self.library.curobo_robot_last_error().decode())

    def forward(self, q):
        request = Request(q.data_ptr(), self.positions.data_ptr(), self.quaternions.data_ptr(),
                          self.jacobians.data_ptr(), torch.cuda.current_stream().cuda_stream, q.shape[0], 1)
        self.check(self.library.curobo_robot_compute_kinematics(self.handle, ct.byref(request)))

    def close(self):
        torch.cuda.synchronize()
        self.library.curobo_robot_destroy(self.handle)


def graph_timing(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    samples = []
    for _ in range(20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / 10)
    return {"median_ms": statistics.median(samples), "p95_ms": sorted(samples)[18]}, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--spheres", required=True, type=Path, help="sphere collision URDF")
    parser.add_argument("--collision-policy", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter-library", type=Path, help="Optional fused free-base/sphere CUDA adapter")
    parser.add_argument("--check-only", action="store_true", help="Skip timings for sanitizer validation")
    parser.add_argument("--reference-output",type=Path,help="Export identical inputs and FP64 oracle for the C++ reference probe")
    parser.add_argument("--nodes", nargs="+", type=int, default=[31, 257, 1025])
    args = parser.parse_args()
    measure = (lambda fn: ({"skipped":True},None)) if args.check_only else graph_timing
    model = pin.buildModelFromUrdf(str(args.urdf)); data = model.createData()
    root_xml = ET.parse(args.spheres).getroot()
    spheres = []
    for link in root_xml.findall("link"):
        for collision in link.findall("collision"):
            shape = collision.find("./geometry/sphere")
            if shape is None:
                raise ValueError("sphere URDF contains non-sphere collision geometry")
            origin = collision.find("origin")
            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            spheres.append((link.attrib["name"], np.array([float(x) for x in xyz.split()]),
                            float(shape.attrib["radius"])))
    manifest = json.loads((args.artifact / "robot_kinematics_manifest.json").read_text())
    tools = manifest["tool_frames"]
    q_indices = [model.joints[model.getJointId(name)].idx_q for name in manifest["joint_names"]]
    v_indices = [model.joints[model.getJointId(name)].idx_v for name in manifest["joint_names"]]
    if sorted(v_indices) != list(range(6, model.nv)) or model.nq != model.nv + 1:
        raise ValueError("probe requires Kyon's free-flyer plus scalar articulated joints")
    reference = np.genfromtxt(args.trajectory, delimiter=",", names=True)
    start = np.array([reference[f"q{i}"][0] for i in range(model.nq)])
    rng = np.random.default_rng(42)
    device = lambda x, dtype=torch.float64: torch.as_tensor(x, dtype=dtype, device="cuda").contiguous()
    offsets = device(np.array([s[1] for s in spheres]))
    radii = device(np.array([s[2] for s in spheres]))
    frame_indices = device([tools.index(s[0]) for s in spheres], torch.long)
    q_map = device(q_indices, torch.long); order = device(np.argsort(v_indices), torch.long)
    base_q_map = device([6,3,4,5], torch.long)
    root = model.jointPlacements[1]; root_R = device(root.rotation); root_t = device(root.translation)
    policy = json.loads(args.collision_policy.read_text())
    disabled = {frozenset((p["link1"], p["link2"])) for p in policy["disabled_pairs"]}
    pairs = [(i,j) for i,a in enumerate(spheres) for j,b in enumerate(spheres)
             if i < j and a[0] != b[0] and frozenset((a[0], b[0])) not in disabled]
    left = device([p[0] for p in pairs], torch.long); right = device([p[1] for p in pairs], torch.long)
    pin.forwardKinematics(model, data, start); pin.updateFramePlacements(model, data)
    obstacle = data.oMf[model.getFrameId(spheres[6][0])].act(spheres[6][1]) + np.array([.09, 0, 0])
    dims = 151; voxel = .04; origin = -3.
    z,y,x = np.meshgrid(np.arange(dims),np.arange(dims),np.arange(dims),indexing="ij")
    grid_np = (np.sqrt((origin+voxel*x-obstacle[0])**2 + (origin+voxel*y-obstacle[1])**2 +
                       (origin+voxel*z-obstacle[2])**2)-.06).astype(np.float32)
    grid = device(grid_np.reshape(-1), torch.float32)
    corners = device([[x,y,z] for z in range(2) for y in range(2) for x in range(2)],torch.long)

    def query(centers):
        p = (centers-origin)/voxel; lo = p.floor().long().clamp(0,dims-2); f = p-lo
        index = lo[...,None,:]+corners
        values = grid[(index[...,2]*dims+index[...,1])*dims+index[...,0]].double()
        w = torch.where(corners.bool(),f[...,None,:],1-f[...,None,:])
        distance = (values*w.prod(-1)).sum(-1)
        signs = (2*corners-1).double()
        spatial = torch.stack([(values*signs[:,a]*w[..., (a+1)%3]*w[..., (a+2)%3]).sum(-1)/voxel
                               for a in range(3)],-1)
        return distance,spatial

    def cost(centers, jac):
        distance, spatial = query(centers)
        clear = distance-radii
        delta = centers[:,left]-centers[:,right]
        length = (delta.square().sum(-1)+1e-24).sqrt()
        self_clear = length-radii[left]-radii[right]
        clear = torch.cat((clear,self_clear),1)
        d_env = torch.einsum('bsa,bsav->bsv',spatial,jac)
        d_self = torch.einsum('bsa,bsav->bsv',delta/length[...,None],jac[:,left]-jac[:,right])
        z = (.02-clear)/.005
        r = torch.nn.functional.softplus(z,threshold=40)*.005
        j = -torch.sigmoid(z)[...,None]*torch.cat((d_env,d_self),1)
        return .5*10000*r.square().sum(1),10000*torch.einsum('bsv,bs->bv',j,r),10000*torch.bmm(j.transpose(1,2),j)

    results = {"precision":"FP32 native FK, FP64 adapter/interpolation/cost/GN", "active_environment_cost":True,
               "timing":"CUDA graph device-only; no upload/download or solver timing", "cases":[]}
    adapter = None
    if args.reference_output:
        args.reference_output.mkdir(parents=True,exist_ok=True)
        grid_np.tofile(args.reference_output/'grid.f32')
        reference_manifest={"urdf":str(args.urdf.resolve()), "sphere_urdf":str(args.spheres.resolve()),
                            "collision_policy":str(args.collision_policy.resolve()), "dimensions":[dims]*3,
                            "origin":[origin]*3,"voxel_size":voxel,"batches":args.nodes}
        (args.reference_output/'manifest.json').write_text(json.dumps(reference_manifest,indent=2)+'\n')
    if args.adapter_library:
        adapter = ct.CDLL(str(args.adapter_library))
        adapter.horizon_curobo_adapt.argtypes = [ct.c_void_p]*11+[ct.c_int]*4+[ct.c_void_p]
    for nodes in args.nodes:
        native = Native(args.library,args.artifact,nodes)
        qs = np.array([pin.integrate(model,start,rng.normal(0,.03,model.nv)) for _ in range(nodes)])
        q = device(qs);qj = q[:,q_map].float().contiguous()

        def adapt():
            native.forward(q[:,q_map].float().contiguous())
            R = rotation(native.quaternions.double())[:,frame_indices]
            offset = (R@offsets[None,...,None]).squeeze(-1)
            local = native.positions[:,frame_indices].double()+offset
            jf = native.jacobians[:,frame_indices].double()
            local_j = (jf[...,:3,:]-skew(offset)@jf[...,3:,:])[...,order]
            base_R = root_R@rotation(q[:,base_q_map])
            base_t = (root_R@q[:,:3,None]).squeeze(-1)+root_t
            centers = (base_R[:,None]@local[...,None]).squeeze(-1)+base_t[:,None]
            J = torch.cat((base_R[:,None].expand(-1,len(spheres),-1,-1),
                           -base_R[:,None]@skew(local),base_R[:,None]@local_j),-1)
            return centers,J

        centers,jac = adapt();torch.cuda.synchronize()
        actual_c,actual_j = centers.cpu().numpy(),jac.cpu().numpy()
        expected_c = np.empty_like(actual_c);expected_j = np.empty_like(actual_j)
        for node,configuration in enumerate(qs):
            pin.computeJointJacobians(model,data,configuration);pin.updateFramePlacements(model,data)
            for i,(name,offset,_) in enumerate(spheres):
                frame = model.getFrameId(name);pose = data.oMf[frame]
                expected_c[node,i] = pose.act(offset)
                j = pin.getFrameJacobian(model,data,frame,pin.LOCAL_WORLD_ALIGNED)
                expected_j[node,i] = j[:3]-pin.skew(pose.rotation@offset)@j[3:]
        if expected_c.min()<origin or expected_c.max()>origin+voxel*(dims-1):
            raise ValueError("probe configuration leaves the finite test SDF")
        actual = cost(centers,jac);expected = cost(device(expected_c),device(expected_j))
        if args.reference_output:
            qs.tofile(args.reference_output/f'q{nodes}.f64')
            c,g,h=[t.cpu().numpy() for t in expected]
            i,j=np.triu_indices(model.nv)
            np.concatenate((c[:,None],g,h[:,i,j]),axis=1).tofile(args.reference_output/f'expected{nodes}.f64')
        errors = {"center_max_abs":float(np.max(np.abs(actual_c-expected_c))),
                  "jacobian_max_abs":float(np.max(np.abs(actual_j-expected_j)))}
        for name,a,b in zip(("cost","gradient","gn_hessian"),actual,expected):
            errors[name+"_max_abs"] = (a-b).abs().max().item()
            errors[name+"_relative_norm"] = ((a-b).norm()/(b.norm()+1e-30)).item()
        if errors['center_max_abs']>2e-5 or errors['jacobian_max_abs']>2e-5:
            raise RuntimeError(f"native free-base adapter failed: {errors}")
        fk_timing,_ = measure(lambda:native.forward(qj))
        adapted_timing,_ = measure(adapt)
        def pipeline():
            centers,jac = adapt();return cost(centers,jac)
        pipeline_timing,_ = measure(pipeline)
        case={"nodes":nodes,"spheres":len(spheres),"self_pairs":len(pairs),"errors":errors,
              "native_fk_jacobian":fk_timing,"fk_sphere_tangent_adapter":adapted_timing,
              "full_collision_prototype":pipeline_timing,"max_cost":actual[0].max().item()}
        if adapter:
            fused_centers=torch.empty_like(centers);fused_jac=torch.empty_like(jac)
            def fused_adapt():
                native.forward(q[:,q_map].float().contiguous())
                tensors=(native.positions,native.quaternions,native.jacobians,q,root_R,root_t,
                         frame_indices,offsets,order,fused_centers,fused_jac)
                status=adapter.horizon_curobo_adapt(*(t.data_ptr() for t in tensors),nodes,
                    len(spheres),len(tools),len(q_indices),torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"fused adapter CUDA error {status}")
                return fused_centers,fused_jac
            fused_adapt();torch.cuda.synchronize()
            center_error=(fused_centers-centers).abs().max().item()
            jac_error=(fused_jac-jac).abs().max().item()
            if center_error>1e-10 or jac_error>1e-10:
                raise RuntimeError(f"fused adapter mismatch: {center_error}, {jac_error}")
            case['fused_adapter_parity_max_abs']={"centers":center_error,"jacobians":jac_error}
            case['native_fk_fused_adapter'],_=measure(fused_adapt)
            def fused_pipeline():
                c,j=fused_adapt();return cost(c,j)
            case['fused_adapter_collision_prototype'],_=measure(fused_pipeline)
        results['cases'].append(case);print(json.dumps(case),flush=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        native.close()


if __name__ == '__main__':
    main()
