import yaml


file = open('config.yaml','r')
lista = file.readlines()
t_min = int(lista[12].split()[1]) + 3*86400
t_max = int(lista[13].split()[1]) + 3*86400
file.close()


stream = open('config.yaml', 'r')
data = yaml.full_load(stream)

data['selection']['tmin'] = t_min
data['selection']['tmax'] = t_max

with open('config.yaml', 'w') as yaml_file:
    yaml_file.write( yaml.dump(data, default_flow_style=False, sort_keys=False))
