import yaml
import re

class YamlParser:

    @classmethod
    def load(cls, yaml_file):

        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(yaml_file, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # parse required fields
        required_fields = ['constraints', 'costs', 'solver'] # 'solver'
        required_fields_dict = cls._parse_required_fields(parsed_yaml, required_fields)

        # parse tasks
        field_names = {'constraints': 'constraint', 'costs': 'residual'}

        # TODO tutto questo fa cagare diocane # refactor in a good way
        task_list = list()
        non_active_task_list = list()
        for task_name, task_desc in parsed_yaml.items():
            task_desc['name'] = task_name

            if task_name in required_fields_dict['costs']:
                task_desc['fun_type'] = 'residual'
                task_list.append(task_desc)
            elif task_name in required_fields_dict['constraints']:
                task_desc['fun_type'] = 'constraint'
                task_list.append(task_desc)
            else:
                non_active_task_list.append(task_desc)


        return task_list, non_active_task_list, required_fields_dict['solver']

    @staticmethod
    def _parse_required_fields(parsed_yaml, required_fields):

        required_fields_dict = dict()
        for field in required_fields:
            if field in parsed_yaml:
                required_fields_dict[field] = parsed_yaml.pop(field)
            else:
                raise ValueError(f"'{field}' field is required.")

        return required_fields_dict


    @staticmethod
    def resolve(task_desc, shortcuts):
        '''
        substitute values in the task description with the corresponding values in a shortcut dictionary
        shortcut: dict of dict ({task_key: {old_value: new_value}})
        '''
        task_desc_copy = task_desc.copy()
        # for each key, value in the task_description
        for k, v in task_desc_copy.items():
            # if the key of the task description is in the dictionary keys
            # if the value of the task description is in the dictionary values,
            # then substitute
            if isinstance(v, str) and k in shortcuts and v in shortcuts[k]:
                task_desc_copy[k] = shortcuts[k][v]
            elif isinstance(v, str) and len(v) >= 3 and v[:2] == '${' and v[-1] == '}':
                try:
                    ret = eval(v[2:-1], {}, {'N': shortcuts['nodes']['final']})
                    task_desc_copy[k] = ret
                except BaseException as e:
                    pass


        return task_desc_copy


if __name__ == '__main__':

    yaml_file = "../playground/spot/task_interface_playground/config_walk.yaml"
    task_list = YamlParser.load(yaml_file)
    for task in task_list:
        print(task)
    shortcuts = {
        'nodes': {'final': [5], 'all': 'thesingleladies'},
        'indices': {'all': 'stars'}
    }
    print("--------------------------------------")
    task_list_new = YamlParser.resolve(task_list[0], shortcuts)

    print(task_list_new)
