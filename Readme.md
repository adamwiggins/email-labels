# Email labeling with models, a holiday hack project

Tinkering around on using language models to sort my emails into Inbox, FYI, and 
Junk.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fetch your Fastmail API token and OpenAI API key from their respective settings 
pages.

To use in the VS Code debugger, copy `.vscode/launch.json.example` to 
`.vscode/launch.json` and fill in the keys.

To run at the terminal, set `FASTMAIL_API_TOKEN` and `OPENAI_API_KEY` environment 
variables before running the scripts.

To use Llama, you can use [oolama.com](https://ollama.dev/) to easily run locally.

## Run

Then, run the scripts in this order:

### 1. `dataset_builder.py`

This connects to fastmail and pulls down some random emails. It'll show you a 
markdown summary. Indicate the label for each. The results will be saved to 
`email_dataset.sqlite`.

Open that in the sqlite console or a visualizer tool like TablePlus. Look through 
to make sure all the data is clean and delete any weird examples. Ideally you want 
50-100 for evals.

When you're satisfied, move `email_dataset.sqlite` to `datasets/for-evals.sqlite`.

### 2. `evals.py`

Now you can test your prompt and model!

Edit the prompt in `classify_email.py` to your taste, and select an LLMProvider 
(`gpt-4o`, `llama3.2`, etc) in `evals.py`.

Run `evals.py` to see how well your model does on accuracy and performance. (TODO: 
add cost.) Make tweaks and try again!

### 3. `fastmail_watcher.py`

Once you've got a model and prompt you're happy with, this will connect to 
Fastmail and watch for new emails. TODO: actually add the label, right now it 
prints it out to the console.

### Fine-tuning

If you want to fine-tune (OpenAI or DistilBERT), you should create a second 
dataset distinct from your eval set.

Run `dataset_builder.py` again. Ideally you want lots here, 100 at least. Also 
helpful is if the labels are balanced, e.g. 33 of each.

When finished, copy `email_dataset.sqlite` to `datasets/for-finetuning.sqlite`.

If you want to fine-tune an OpenAI model, run `finetune_openai.py` to convert 
your SQLite file into a JSONL file. You can then upload that to the OpenAI 
dashboard to create a new fine-tuning job.

If you want to fine-tune a DistilBERT classifier, run 
`providers/distilbert_provider.py`. TODO: I'm not sure this really works, the 
resulting accuracy for me was poor, so likely my very limited ML knowledge has me 
missing something important.

In both cases you can check the results as usual with an eval, by updating the 
LLMProvider in `evals.py`.

## License

MIT I guess, surely you aren't actually going to use this code for anything :)
