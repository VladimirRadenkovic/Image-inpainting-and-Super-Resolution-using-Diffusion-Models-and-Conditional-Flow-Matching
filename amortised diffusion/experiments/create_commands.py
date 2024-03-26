import os
from itertools import product


class CommandsBuilder:
    r"""
    Creates the outer-product of configurations to be executed.
    Returns a list with all the combinations.
    Here's an example:
    ```
    commands = (
        CommandsBuilder()
        .add("dataset", ["Power", "Kin8mn"])
        .add("split", [0, 1])
        .build()
    )
    ```
    Returns
    ```
    commands = [
        "python main.py with dataset=Power split=0;",
        "python main.py with dataset=Power split=1;",
        "python main.py with dataset=Kin8mn split=0;",
        "python main.py with dataset=Kin8mn split=1;",
    ]
    ```
    """
    command_template = """
python main.py --mode=eval --save_dir=logs_eval --config=config.py:{dataset},{task},{method} {config};"""
    single_config_template = " --{key}={value}"

    def __init__(self, dataset, task, method) -> None:
        self.dataset, self.task, self.method = dataset, task, method
        self.keys = []
        self.values = []

    def add(self, key, values):
        self.keys.append(key)
        self.values.append(values)
        return self

    def build(self):
        commands = []
        for args in product(*self.values):
            config = ""
            for key, value in zip(self.keys, args):
                config += self.single_config_template.format(key=key, value=value)
            command = self.command_template.format(
                dataset=self.dataset,
                task=self.task,
                method=self.method,
                config=config
            )
            commands.append(command)
        return commands
    

DATASETS = [
    "flowers",
    "celeba",
]

METHODS = [
    "amortized",
    "reconstruction_guidance",
    "replacement",
]

if __name__ == "__main__":
    NAME = "commands_eval.txt"

    if os.path.exists(NAME):
        print("File to store script already exists", NAME)
        exit(-1)

    commands = []
    for dataset in DATASETS:
        for method in METHODS:
            commands.extend(
                CommandsBuilder(dataset=dataset, task="outpainting", method=method)
                .add("config.testing.seed", range(5))
                .build()
            )

    with open(NAME, "w") as file:
        file.write("".join(commands))