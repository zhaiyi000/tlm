from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
import random
import copy
import pickle


@dataclass
class ScriptArguments:
    # device: str = field(default="cuda:0", metadata={"help": ""})
    pass


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx][0]), torch.tensor(self.data_list[idx][1])


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(train_data_list, eval_data_list, model_path, input_size, hidden_size, output_size, batch_size, epochs, learning_rate, device):
    train_dataset = MyDataset(train_data_list)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = MyDataset(eval_data_list)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    step = 0
    total_step = epochs * len(train_dataloader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lambda1 = lambda step: 1 - step / total_step / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    writer = SummaryWriter(log_dir=os.path.join(model_path, 'runs'))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (features, labels) in enumerate(train_dataloader):
            step += 1
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # evaluation
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for i, (features, labels) in enumerate(eval_dataloader):
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                eval_loss += loss.item()
        eval_loss /= len(eval_dataloader)

        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, Learning_rate: {lr}')

        # write to tensorboard
        writer.add_scalars('Loss', {'train':train_loss, 'eval':eval_loss}, epoch+1)
        writer.add_scalars('lr', {'lr': lr}, epoch+1)

        # save model every epoch
    torch.save(model.state_dict(), os.path.join(model_path, f'model_epoch_{epochs}.pth'))

    writer.close()


def read_gen_best_json(json_path):
    with open(json_path, "r") as f:
        lines = f.read().strip().split("\n")
    min_latency_dict = {}
    for line in lines:
        json_line = json.loads(line)
        latency = json_line["latency"]
        json_line = json.loads(json_line["line"])
        workload_key = json_line["i"][0][0]
        if workload_key not in min_latency_dict:
            min_latency_dict[workload_key] = 1e10
        min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
    return min_latency_dict


def read_fine_tuning_json(json_path):
    with open(json_path, "r") as f:
        lines = f.read().strip().split("\n")
    min_latency_dict = {}
    for line in lines:
        json_line = json.loads(line)
        latencies = json_line["r"][0]
        latency = sum(latencies) / len(latencies)
        workload_key = json_line["i"][0][0]
        if workload_key not in min_latency_dict:
            min_latency_dict[workload_key] = 1e10
        min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
    return min_latency_dict


def slog(it):
    it[it<0] = 0
    return np.log(it + 1e-8)


def get_mean_std(data_list):
    data_list = np.concatenate(data_list)
    data_list = slog(data_list)
    data_mean, data_std = np.mean(data_list), np.std(data_list)
    return data_mean, data_std


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    max_len = 50
    hidden_size = 256
    output_size = 1
    batch_size = 32
    learning_rate = 2e-4
    epochs = 100
    device = 'cuda:0'

    best_history = {}
    min_latency_dict_1 = read_gen_best_json("gen_data/finetuning_0.json")
    for key, val in min_latency_dict_1.items():
        best_history[key] = [val]

    from gen_utils import get_finetuning_files
    finetuning_list = get_finetuning_files()
    for file in finetuning_list:
        min_latency_dict_2 = read_fine_tuning_json(file)
        for key, val in min_latency_dict_2.items():
            best_history[key].append(val)
    best_history_list = list(best_history.items())

    workload_key_list = []
    for key, val in best_history_list:
        workload_key_list.append(eval(key)[0])
    workload_key_list.sort()

    data_list = []
    for key, hisotry in best_history_list:
        data = eval(key)
        data[0] = workload_key_list.index(data[0])
        if len(hisotry) == 1:
            hisotry.append(0)
        data.extend(hisotry)
        assert(len(data) <= max_len)
        data_list.append(np.array(data, dtype=np.float32))
    data_mean, data_std = get_mean_std(data_list)

    train_val_data_list = []
    test_data_list = []
    for key, hisotry in best_history_list:
        data = eval(key)
        data[0] = workload_key_list.index(data[0])
        if len(hisotry) == 1:
            hisotry.append(0)

        data_list = []
        for i in range(1, len(hisotry) + 1, 1):
            data_i = copy.deepcopy(data)
            for ii in range(i):
                data_i.append(hisotry[ii])
            if i < len(hisotry):
                data_i.append(hisotry[i-1] - hisotry[i])
                data_i = slog(np.array(data_i, dtype=np.float32))
                data_i = (data_i - data_mean) / data_std
                label = data_i[-1]
                data_i = data_i[:-1]
                inputs = np.pad(data_i, (0, max_len - len(data_i)), 'constant')
                train_val_data_list.append((inputs, label))
            else:
                data_i = slog(np.array(data_i, dtype=np.float32))
                data_i = (data_i - data_mean) / data_std
                inputs = np.pad(data_i, (0, max_len - len(data_i)), 'constant')
                test_data_list.append(inputs)
    
    random.shuffle(train_val_data_list)
    print('len train_val_data_list', len(train_val_data_list))
    train_data_list = train_val_data_list[:int(len(train_val_data_list) * 0.9)]
    eval_data_list = train_val_data_list[int(len(train_val_data_list) * 0.9):]
    
    train_model(train_data_list, eval_data_list, model_path, max_len, hidden_size, output_size, batch_size, epochs, learning_rate, device)

    find_potential_keys(test_data_list, best_history_list, max_len, hidden_size, output_size, device, epochs)


def find_potential_keys(test_data_list, best_history_list, max_len, hidden_size, output_size, device, epochs):
    model = MLP(max_len, hidden_size, output_size).to(device)
    state_dict = torch.load(os.path.join(model_path, f'model_epoch_{epochs}.pth'))
    model.load_state_dict(state_dict)

    score_list = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(test_data_list), batch_size):
            inputs = test_data_list[start : start + batch_size]
            inputs = torch.tensor(inputs).to(device)
            preds = model(inputs)
            score_list.append(preds.squeeze().detach().cpu())
    scores = torch.cat(score_list)
    want_cnt = int(len(best_history_list) / 4)
    indices = scores.topk(want_cnt).indices.tolist()

    from common import clean_name
    key_str_set = set()
    for idx in indices:
        key, val = best_history_list[idx]
        key_str_set.add(clean_name((key, "llvm")))

    with open(os.path.join(model_path, 'task_sheduler.pkl'), 'wb') as f:
        pickle.dump(key_str_set, f)


model_path = 'gen_data/task_sheduler'
if __name__ == "__main__":
    main()
    # find_potential_files()


def find_potential_files(files):
    with open(os.path.join(model_path, 'task_sheduler.pkl'), 'rb') as f:
        key_str_set = pickle.load(f)

    potential_files = []
    for file in files:
        if os.path.splitext(os.path.basename(file))[0] in key_str_set:
            potential_files.append(file)
    return potential_files