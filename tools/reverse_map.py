import jsonlines
import pickle
import copy
import sys
import os
import shutil

TASK_NAME = "ball_in_hoop_temporal"
data_map_path = f"/home/local/ASUAD/opatil3/datasets/{TASK_NAME}/dataset/data_map.jsonl"
data_path = f"/home/local/ASUAD/opatil3/datasets/{TASK_NAME}/dataset/model_results/"
# data_map_path = (
#     "/home/local/ASUAD/opatil3/datasets/stack_blocks_temporal/dataset/data_map.jsonl"
# )
# data_path = "/home/local/ASUAD/opatil3/datasets/stack_blocks_temporal/dataset/json/"
fps = 10
data_map = {}
reader = jsonlines.open(data_map_path)
for obj in reader:
    data_map[obj["auto_id"]] = obj["data_path"]

reader = []
for s in os.listdir(data_path):
    z = jsonlines.open(data_path + s)
    for obj in z:
        reader.append(obj)
print(len(reader))
negate_map = {}


def classify_task(query, inst):
    index = -1
    for instructions in inst:
        for i, instruction in enumerate(instructions):
            if instruction == query:
                index = i
                break

    if index == -1:
        return "test"
    else:
        return inst[-1][index]


def classify_task_1(query, inst):
    index = -1
    for instructions in inst:
        for i, instruction in enumerate(instructions):
            if instruction == query:
                index = i
                break
    # print(query, index)
    if index == 5:
        return inst[-1][index]
    else:
        return None


def copy_data(query, demo, start, end, source, destination):
    # Task names = "Go to", "Pick", "Place", "Open" etc.
    # variations = "different tasks"
    # eposides = all variation from a task

    if not os.path.exists(destination):
        os.makedirs(destination)

    new_demo = copy.deepcopy(demo)

    folders = [name for name in os.listdir(source) if name != "low_dim_obs.pkl"]

    def get_chunk(self, query, start, end):
        self._observations = self._observations[start : end + 1]
        self.change_point = self.change_point[start : end + 1]
        index = -1
        for instructions in self.instructions:
            for i, instruction in enumerate(instructions):
                if instruction == query:
                    index = i
                    break
        self.instructions = [
            [instructions[index]] for instructions in self.instructions
        ]
        return self

    new_demo = get_chunk(new_demo, query, start, end)

    xyz = classify_task(query, demo.instructions)
    if xyz is None:
        return

    final_path = f"{destination}/{xyz}/{source.split('/')[-4]}/episodes/"
    try:
        episode = sorted(
            [int(name.replace("episode", "")) for name in os.listdir(final_path)]
        )
        episode = "episode" + str(episode[-1] + 1)
    except:
        episode = "episode0"
    final_path += episode

    os.makedirs(final_path)

    with open(os.path.join(final_path, "low_dim_obs.pkl"), "wb") as openfile:
        pickle.dump(new_demo, openfile)

    for folder in folders:
        dest_folder = os.path.join(final_path, folder)
        os.makedirs(dest_folder)
        for index in range(start, end + 1):
            try:
                shutil.copy(
                    f"{source}/{folder}/{index}.png",
                    f"{dest_folder}/{index - start}.png",
                )
            except:
                pass


cntr = 0
for obj in reader:
    try:
        demo = pickle.load(open(f"{data_map[ obj['vid'] ]}low_dim_obs.pkl", "rb"))
        duration = obj["duration"] * fps
        for [start, end] in obj["relevant_windows"]:
            # print(start, end)
            start *= fps
            end *= fps
            if (obj["vid"], start, end) in negate_map.keys():
                negate_map[(obj["vid"], start, end)] += 1
                continue
            else:
                negate_map[(obj["vid"], start, end)] = 0
            negate_map[(obj["vid"], start, end)] += 1
            copy_data(
                obj["query"],
                demo,
                int(start),
                int(end),
                data_map[obj["vid"]],
                f"/home/local/ASUAD/opatil3/datasets/{TASK_NAME}/withheld_task_data",
            )
    except Exception as e:
        print("exception", e)
        cntr += 1
print(cntr)
print(negate_map)
