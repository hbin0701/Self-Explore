import sys
import json
from datasets import Dataset, DatasetDict


def get_dataset(path_train, path_test):
    
    final_dict = {}

    for idx, path in enumerate([path_train, path_test]):
        li = [json.loads(x) for x in open(path)]

        text_prompts = []
        chosen = []
        rejected = []

        # If test (i.e. validation set), just use dummy data: subset of training.
        if idx == 1:
            li = li[:100] 

        for elem in li:

            if 'prompt' not in elem.keys():
                elem['prompt'] = elem['input']

            text_prompts.append(elem['prompt'].rstrip("\n") + "\n")
            chosen.append(elem['chosen'])

            rej = elem['rejected']

            # [Note] Remove last line for rejected sample.
            tgt_string = "The answer is"
            if tgt_string in rej and len(rej.split("\n")) > 1:
                rej = rej[:rej.index(tgt_string)].strip()

            rejected.append(rej)
        
        d = {"prompt": text_prompts, "text_prompt": text_prompts, "chosen": chosen, "rejected": rejected}
        
        if idx == 0:
            final_dict["train"] = d
        else:
            final_dict["test"] = d

    train_dataset = Dataset.from_dict(final_dict["train"])
    test_dataset = Dataset.from_dict(final_dict["test"])

    print("train example", train_dataset[0])

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset.shuffle(seed=42),
        'test': test_dataset.shuffle(seed=42)
    })

    return dataset_dict