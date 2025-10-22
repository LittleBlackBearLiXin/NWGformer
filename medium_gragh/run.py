import subprocess

def run_experiment(data, model):
    # Check which model and dataset are selected, then run the appropriate command
    if data == 'amazon-computer' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln"
    elif data == 'amazon-computer' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset amazon-computer --hidden_channels 256 --epochs 1000 --lr 0.002 --runs 5 --local_layers 1 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln"

    elif data == 'amazon-photo' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 5 --local_layers 1 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --res"
    elif data == 'amazon-photo' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device 0 --ln --res"

    elif data == 'coauthor-cs' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device 0 --ln --res"
    elif data == 'coauthor-cs' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --epochs 200 --lr 0.0003 --runs 5 --local_layers 1 --weight_decay 1e-3 --dropout 0.5 --device 0 --ln --res"

    elif data == 'coauthor-physics' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 5 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device 0 --ln --res"
    elif data == 'coauthor-physics' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset coauthor-physics --hidden_channels 300 --epochs 100 --lr 0.0005 --runs 5 --local_layers 1 --weight_decay 1e-3 --dropout 0.7 --device 0 --bn --res"

    elif data == 'amazon-ratings' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 5 --local_layers 1 --weight_decay 0.0 --dropout 0.5 --device 0 --bn --res"
    elif data == 'amazon-ratings' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device 0 --bn --res"

    elif data == 'minesweeper' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 5 --local_layers 11 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 0 --bn --res"
    elif data == 'minesweeper' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 5 --local_layers 18 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device 0 --bn --res"

    elif data == 'roman-empire' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 5 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device 0 --bn --res"
    elif data == 'roman-empire' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 5 --local_layers 9 --weight_decay 0.0 --dropout 0.3 --device 0 --bn --res"

    elif data == 'questions' and model == 'gcn':
        command = "/root/miniconda3/bin/python main.py --gnn gcn --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 5 --local_layers 9 --weight_decay 0.0 --dropout 0.3 --metric rocauc --device 0 --res"
    elif data == 'questions' and model == 'gat':
        command = "/root/miniconda3/bin/python main.py --gnn gat --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 1e-4 --runs 5 --local_layers 1 --weight_decay 0.0 --dropout 0.3 --metric rocauc --device 0 --ln --res"

    else:
        print(f"Invalid combination: {data} and {model}")
        return

    # Run the command using subprocess
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Example usage: Run for 'amazon-computer' dataset and 'gcn' model
    dataset_name = 'questions'  # Replace with dynamic input
    model_name = 'gat'  # Replace with dynamic input
    run_experiment(dataset_name, model_name)

