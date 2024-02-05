import jsonlines
import pickle
import copy
import sys
import os
import shutil

data_map_path = "/home/local/ASUAD/draj5/dataset/data_map.jsonl"
data_path = sys.argv[1]

fps = 10
data_map = {}
reader = jsonlines.open (data_map_path)
for obj in reader:
    data_map[ obj['auto_id'] ] = obj['data_path']


reader = jsonlines.open (data_path)
negate_map = {}

def classify_task (query, inst):
    index = -1
    for instructions in inst:
        for i, instruction in enumerate (instructions):
            if instruction == query:
                index = i
                break
    
    if index == -1:
        return "test"
    else:
        return inst[-1][index]

def copy_data (query, demo, start, end, source, destination):
    # Task names = "Go to", "Pick", "Place", "Open" etc.
    # variations = "different tasks"
    # eposides = all variation from a task

    if not os.path.exists (destination):
        os.makedirs (destination)

    new_demo = copy.deepcopy (demo)
    
    folders = [name for name in os.listdir (source) if name != 'low_dim_obs.pkl']
    def get_chunk(self, query, start, end):
        self._observations = self._observations[start: end + 1]
        self.change_point = self.change_point[start: end + 1]
        index = -1
        for instructions in self.instructions:
            for i, instruction in enumerate (instructions):
                if instruction == query:
                    index = i
                    break
        self.instructions = [[instructions[index]] for instructions in self.instructions]    
        return self
    new_demo = get_chunk (new_demo, query, start, end)
    
    final_path = f"{destination}/{classify_task(query, new_demo.instructions)}/{source.split('/')[-4]}/episodes/"
    try:
        episode = sorted ([int(name.replace('episode', '')) for name in os.listdir (final_path)])
        episode = 'episode' + str (episode[-1] + 1)
    except:
        episode = 'episode0'
    final_path += episode

    os.makedirs (final_path)

    with open (os.path.join (final_path, 'low_dim_obs.pkl'), 'wb') as openfile:
        pickle.dump (new_demo, openfile)

    for folder in folders:
        dest_folder = os.path.join (final_path, folder)
        os.makedirs (dest_folder)
        for index in range(start, end + 1):
            try:
                shutil.copy (f"{source}/{folder}/{index}.png", f"{dest_folder}/{index - start}.png")
            except:
                pass
    

cntr = 0
for obj in reader:
    try:
        demo = pickle.load (open (f"{data_map[ obj['vid'] ]}low_dim_obs.pkl", 'rb'))
        duration = obj['duration'] * fps
        for [start, end] in obj['relevant_windows']:
            start *= fps
            end *= fps
            if (obj['vid'], start, end) in negate_map.keys ():
                negate_map[(obj['vid'], start, end)] += 1
                continue
            else:
                negate_map[(obj['vid'], start, end)] = 0
            negate_map[(obj['vid'], start, end)] += 1
            copy_data (obj['query'], demo, int(start), int(end), data_map[ obj['vid'] ], "/home/local/ASUAD/draj5/task_data")
    except Exception as e:
        print (e)
        cntr += 1
print (cntr)
print (negate_map)