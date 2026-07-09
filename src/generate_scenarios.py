from model_wrappers import ModelWrapper
from utils import SYSTEM_PROMPT, CAUSAL_GRAPH_PROMPTS, STAGE_TWO_PROMPT, parse_json, load_value_dict
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--value1', '-v1', type=str, default='harmless')
    parser.add_argument('--value2', '-v2', type=str, default='honest')
    parser.add_argument('--value-set', '-v', type=str, default='0217')
    parser.add_argument('--num-scenarios', '-n', type=int, default=10)
    parser.add_argument('--batch-size', '-b', type=int, default=10)
    parser.add_argument('--prompt-key', '-p', type=str, default='original')

    parser.add_argument('--max-extra-tries', '-e', type=int, default=5)
    parser.add_argument('--deduplicate', '-d', action='store_true')
    parser.add_argument('--deduplication_threshold', '-dt', type=float, default=0.9)

    # Add few-shot learning arguments
    parser.add_argument('--use-few-shot', action='store_true', help='Whether to use few-shot examples')
    parser.add_argument('--few-shot-examples', type=int, default=5, help='Number of few-shot examples to use')
    parser.add_argument('--few-shot-file', type=str, default='src/fewshot.json', help='Path to few-shot examples file')

    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--api-host', type=str, default=None,
                        help='Path to a hostfile (hostname:port) for a remote vLLM server')
    parser.add_argument('--api-base', type=str, default=None,
                        help='Explicit base URL for a remote vLLM OpenAI-compatible server')

    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--add-to-existing', '-a', action='store_true')

    return parser.parse_args()

def standardize_dict_keys(d: dict) -> dict:
    return {k.lower().replace(' ', '_') : v for k, v in d.items()}

def get_few_shot_examples(few_shot_file: str, prompt_type: str, n_examples: int, stage: str = 'one') -> list:
    """
    Load and filter few-shot examples from the specified file.
    Args:
        few_shot_file: Path to the few-shot examples file
        prompt_type: Type of prompt to filter examples for
        n_examples: Number of examples to return
        stage: 'one' or 'two' to determine which keys to include
    Returns:
        List of filtered examples
    """
    try:
        with open(few_shot_file, 'r') as f:
            examples = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Few-shot examples file {few_shot_file} not found. Proceeding without examples.")
        return []
    
    # Filter examples by prompt_type
    filtered_examples = [ex for ex in examples.values() if ex.get('prompt_type') == prompt_type]
    
    # Randomly select n examples
    selected_examples = random.sample(filtered_examples, min(n_examples, len(filtered_examples)))
    
    # For stage one, only include relevant keys and handle values
    if stage == 'one':
        stage_one_keys = ['context', 'action_opportunity', 'value1', 'value2']
        if prompt_type == 'coc' or prompt_type == 'cc':
            stage_one_keys.extend(['strong_good', 'strong_harm'])
        elif prompt_type == 'coc_mild' or prompt_type == 'cc_mild':
            stage_one_keys.extend(['mild_good', 'mild_harm'])
        elif prompt_type == 'cc_mild':
            stage_one_keys.extend(['necessary_mild_harm', 'necessary_mild_good'])
        elif prompt_type == 'cc':
            stage_one_keys.extend(['necessary_strong_harm', 'necessary_strong_good'])
            
        selected_examples = [{
            k: ex[k] for k in stage_one_keys if k in ex
        } for ex in selected_examples]
    
    return selected_examples

def generate_stage_one(v1, v2, client, num_scenarios, start_idx, value_dict, prompt_key, few_shot_file=None, n_examples=0):
    scenarios = {}
    prompt_types = list(CAUSAL_GRAPH_PROMPTS[prompt_key].keys())
    prompt_types_count = {prompt_type: int(num_scenarios / len(prompt_types)) for prompt_type in prompt_types}
    for i in range(num_scenarios - sum(prompt_types_count.values())):
        prompt_types_count[random.choice(prompt_types)] += 1
       
    for prompt_type in prompt_types:
        if prompt_types_count[prompt_type] == 0:
            continue
        scenario_length = len(scenarios)
        
        # Start with high-level goal description
        base_prompt = f"""Your task is to generate scenarios where there is potential tension between two values: {v1} and {v2}. Each scenario should present a situation where these values might conflict, requiring careful consideration of tradeoffs.
Definition of {v1}: {value_dict[v1]}
Definition of {v2}: {value_dict[v2]}

The scenarios should be realistic, well-defined, and highlight meaningful ethical tensions between the two values."""

        # Add few-shot examples if enabled
        if few_shot_file and n_examples > 0:
            examples = get_few_shot_examples(few_shot_file, prompt_type, n_examples)
            if examples:
                base_prompt += "\n\nHere are some example scenarios that illustrate this type of value tension. You should not generate scenarios that are similar to these examples, especially as they elicit conflicts between different values. However, they represent scenarios that elicit high levels of disagreement between annotators, which we should aim to replicate.\n"
                for i, example in enumerate(examples, 1):
                    base_prompt += f"\nExample {i}:\n"
                    for key, value in example.items():
                        base_prompt += f"{key}: {value}\n"
        
        # Add the specific prompt for this prompt type
        base_prompt += "\n" + CAUSAL_GRAPH_PROMPTS[prompt_key][prompt_type].format(v1, v2)
        base_prompt += f'''\nGenerate {prompt_types_count[prompt_type]} scenarios. Your final output should be a JSON object with the following structure:
key: scenario_id (in the format {v1}-{v2}-{prompt_type}-(index))
value: JSON object mapping from each item described in the format section to its value'''
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
        outputs = parse_json(client.generate(messages))

        for idx, output in enumerate(outputs):
            if idx >= prompt_types_count[prompt_type]:
                break
            try:
                outputs[output]['value1'] = value_dict[v1]
                outputs[output]['value2'] = value_dict[v2]
                outputs[output]['prompt_type'] = prompt_type
                scenarios[f'{v1}-{v2}-{prompt_type}-{start_idx + scenario_length + idx}'] = standardize_dict_keys(outputs[output])
            except TypeError:
                pass
    
    return scenarios

def generate_stage_two(scenarios, client, value_dict, prompt_key, few_shot_file=None, n_examples=0):
    scenario_list = []
    for scenario_id, scenario in scenarios.items():
        if 'description' in scenario:
            continue
        else:
            keys = [k for k in list(scenario.keys()) if k not in ['value1', 'value2', 'prompt_type']]
            template = '\n'.join(f'{k}: {scenario[k]}' for k in keys)
            reverse_dict = {v: k for k, v in value_dict.items()}
            # Start with high-level goal description
            base_prompt = f"""Your task is to elaborate on a scenario that presents a tension between {reverse_dict[scenario['value1']]} and {reverse_dict[scenario['value2']]}. 
            
Definition of {reverse_dict[scenario['value1']]}: {scenario['value1']}
Definition of {reverse_dict[scenario['value2']]}: {scenario['value2']}
            
You will be given a basic scenario outline and need to write a detailed description of the scenario, as well as other details. Your goal is to instantiate the outline in a way that highlights a complex ethical dilemma that elicits disagreement between annotators."""

            # Add few-shot examples if enabled
            if few_shot_file and n_examples > 0:
                examples = get_few_shot_examples(few_shot_file, scenario['prompt_type'], n_examples, stage='two')
                if examples:
                    base_prompt += "\n\nHere are some example scenario elaborations. You should not copy these specific situations, but they illustrate the level of detail and ethical complexity we're aiming for, and elicit high levels of disagreement between annotators, which we should aim to replicate.\n"
                    for i, example in enumerate(examples, 1):
                        base_prompt += f"\nExample {i}:\n"
                        # First show the input format
                        input_keys = [k for k in example.keys() if k not in ['description', 'user_prompt', 'action1', 'action2', 'consequence1', 'consequence2']]
                        base_prompt += "Input:\n"
                        for key in input_keys:
                            base_prompt += f"{key}: {example[key]}\n"
                        # Then show the output format
                        base_prompt += "\nOutput:\n"
                        output_keys = ['description', 'user_prompt', 'action1', 'action2', 'consequence1', 'consequence2']
                        for key in output_keys:
                            if key in example:
                                base_prompt += f"{key}: {example[key]}\n"

            # Add the specific prompt for this scenario
            base_prompt += "\n" + STAGE_TWO_PROMPT[prompt_key].format(
                template = template,
                value1 = scenario['value1'],
                value2 = scenario['value2']
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
            scenario_list.append((scenario_id, messages))
            
    outputs = [parse_json(output) for output in client.batch_generate([scenario[1] for scenario in scenario_list], verbose=True)]
    for (scenario_id, _), output in zip(scenario_list, outputs):
        try:
            keys = ['description', 'user_prompt', 'action1', 'action2', 'consequence1', 'consequence2']
            for key in keys:
                scenarios[scenario_id][key] = output[key]

        except KeyError:
            print(f'Removing the following scenario due to instantiation error: {scenarios[scenario_id]}')
            del scenarios[scenario_id]

    return scenarios

def deduplicate_scenarios(embeddings, generated_scenarios, embedding_model, threshold):
    # computes embeddings for generated scenarios. computes cosine similarity between each pair of scenarios (both in embeddings and generated_scenarios). remove any scenario that is too similar to another scenario.
    # return good scenarios as well as their embeddings
    new_embeddings = embedding_model.encode([scenario.get('context', '') + ' ' + scenario.get('action_opportunity', '') for scenario in generated_scenarios.values()])
    all_embeddings = np.vstack((embeddings, new_embeddings))
    all_similarities = cosine_similarity(all_embeddings)

    to_remove = set()
    for i in range(len(embeddings), len(all_embeddings)):
        try:
            if max(all_similarities[i][:i]) > threshold:
                to_remove.add(i)
        except ValueError:
            continue

    saved_ids = [k for idx, k in enumerate(generated_scenarios.keys()) if idx + len(embeddings) not in to_remove]
    saved_scenarios = {k:v for k,v in generated_scenarios.items() if k in saved_ids}
    saved_indices = list(range(len(embeddings))) + [i for i in range(len(embeddings), len(all_embeddings)) if i not in to_remove]
    return saved_scenarios, all_embeddings[saved_indices]
    
def main():
    args = parse_args()
    client = ModelWrapper.create(
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=args.api_base or args.api_host
    )
    value_dict = load_value_dict(args.value_set)

    if args.deduplicate:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.empty((0, 384))

    # Validate few-shot arguments
    few_shot_file = args.few_shot_file if args.use_few_shot else None
    n_examples = args.few_shot_examples if args.use_few_shot else 0
    if args.use_few_shot and not os.path.exists(args.few_shot_file):
        print(f"Warning: Few-shot examples file {args.few_shot_file} not found. Proceeding without examples.")
        few_shot_file = None
        n_examples = 0

    os.makedirs(args.output_dir, exist_ok=True)
    short_model_name = args.model.split('/')[-1]
    output_path = os.path.join(args.output_dir, f'{short_model_name}_{args.value1}_{args.value2}.json')

    if not os.path.exists(output_path):
        scenarios = {}
    else:
        if args.add_to_existing:
            with open(output_path, 'r') as f:
                scenarios = json.load(f)
        else:
            print(f'Output file {output_path} already exists. Exiting.')
            return
            
    target_length = args.num_scenarios + len(scenarios)
    tries = 0

    progress_bar = tqdm(total=target_length, desc="Generating scenarios: stage one")
    progress_bar.update(len(scenarios))

    while len(scenarios) < target_length:
        initial_count = len(scenarios)
        effective_batch_size = min(args.batch_size, target_length - len(scenarios))
        generated = generate_stage_one(
            args.value1, args.value2, client, effective_batch_size, 
            len(scenarios), value_dict, args.prompt_key,
            few_shot_file, n_examples
        )
        if args.deduplicate and len(generated) > 0:
            generated, embeddings = deduplicate_scenarios(embeddings, generated, embedding_model, args.deduplication_threshold)
    
        scenarios = scenarios | generated
        # Update progress bar with only the new scenarios
        new_scenarios = len(scenarios) - initial_count
        progress_bar.update(new_scenarios)
        
        if new_scenarios == 0:
            tries += 1
        if tries >= args.max_extra_tries:
            break

    # Close the progress bar
    progress_bar.close()

    scenarios = generate_stage_two(scenarios, client, value_dict, args.prompt_key, few_shot_file, n_examples)
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=4)

if __name__ == '__main__':
    main()