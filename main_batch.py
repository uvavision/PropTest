import os
import pathlib
from functools import partial
import warnings
import traceback

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory

from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from utils import seed_everything

mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)
timeout_duration = 120


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def run_program_wo_VLM(parameters, queues_in_, input_type_):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, process_guesses
    from video_segment import VideoSegment

    global queue_results

    if config.codex.testcase:
        code, sample_id, image, possible_answers, query, test_code, eval_test = parameters  # when eval_test = 1, image is actually answer
        test_code = test_code.replace("solve_query(image)",
                                      "execute_command(image, my_fig, time_wait_between_lines, syntax)")
    else:
        code, sample_id, image, possible_answers, query = parameters
        eval_test = False

    ###### Preprocess codes
    code_header = f'def execute_command_{sample_id}(' \
                    f'{input_type_}, possible_answers, query, ' \
                    f'ImagePatch, VideoSegment, ' \
                    'llm_query, bool_to_yesno, distance, best_image_match, process_guesses):\n' \
                    f'    # Answer is:'

    if config.codex.testcase and not eval_test:
        code_h = f'def execute_command(' \
                f'{input_type_}, possible_answers, query, ' \
                f'ImagePatch, VideoSegment, ' \
                'llm_query, bool_to_yesno, distance, best_image_match, process_guesses):\n' \
                f'    # Answer is:'
        all_testcode = code_header + test_code
        all_testcode = all_testcode.replace('execute_command(image, my_fig, time_wait_between_lines, syntax)',
                                                code_h[4:-18])
        try:
            all_code = code_h + code + '\n' + all_testcode
        except:
            print(f'all_testcode: {all_testcode}')
            print(f'code: {code}')
            print(f'code_h: {code_h}')
    else:
            code_onestep = code_header + code


    try:
        if config.codex.testcase:
            exec(compile(all_code, 'Codex', 'exec'), globals())
        else:
            exec(compile(code_onestep, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        return None, [code]

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)
    process_guesses_partial = partial(process_guesses, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match, process_guesses_partial)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        print(f'Sample {sample_id} failed with error: {e}.')
        return None, [code]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, [code]

def run_program(parameters, queues_in_, input_type_, retrying=False, codes=None):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, process_guesses
    from video_segment import VideoSegment

    global queue_results

    trial = 0
    eval_test_pass = 1  # 0 : assertion error, 1: pass, 2: compilation error

    if config.codex.testcase:
        if len(parameters) == 8:
            code, sample_id, image, possible_answers, query, test_code, eval_test, crop  = parameters
        else:
            code, sample_id, image, possible_answers, query, test_code, eval_test = parameters
        test_code = test_code.replace("solve_query(image)", "execute_command(image, my_fig, time_wait_between_lines, syntax)")
    else:
        code, sample_id, image, possible_answers, query = parameters
        eval_test = False

    generated_codes = [code]

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)
    process_guesses_partial = partial(process_guesses, queues=queues)

    ###### Preprocess codes
    code_header = f'def execute_command_{sample_id}(' \
                    f'{input_type_}, possible_answers, query, ' \
                    f'ImagePatch, VideoSegment, ' \
                    'llm_query, bool_to_yesno, distance, best_image_match, process_guesses):\n' \
                    f'    # Answer is:'

    if config.codex.testcase and not eval_test:
        code_h = f'def execute_command(' \
                 f'{input_type_}, possible_answers, query, ' \
                 f'ImagePatch, VideoSegment, ' \
                 'llm_query, bool_to_yesno, distance, best_image_match, process_guesses):\n' \
                 f'    # Answer is:'
        all_testcode = code_header + test_code
        all_testcode = all_testcode.replace('execute_command(image, my_fig, time_wait_between_lines, syntax)',
                                            code_h[4:-18])
        try:
            all_code = code_h + code + '\n' + all_testcode
        except:
            print(f'all_testcode: {all_testcode}')
            print(f'code: {code}')
            print(f'code_h: {code_h}')
    elif config.codex.testcase and eval_test:
        all_testcode = code_header + test_code
        if config.dataset.dataset_name in ['RefCOCO', 'RefCOCO+']:
            all_code = all_testcode.replace('execute_command(image, my_fig, time_wait_between_lines, syntax)',
                                            f" ImagePatch(image,left={int(crop[0])},lower={int(crop[1])},right={int(crop[2])},upper={int(crop[3])})")
        else:
            all_code = all_testcode.replace('execute_command(image, my_fig, time_wait_between_lines, syntax)',
                                            f"'{image}'")
    else:
        code_onestep = code_header + code

        # Compile the code
    try:
        if config.codex.testcase:
            exec(compile(all_code, 'Codex', 'exec'), globals())
        else:
            exec(compile(code_onestep, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            print(f'Using BLIP2 for sample {sample_id}')
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the BLIP2 worked. Sample {sample_id} failed at compilation time with error: {e2}')
            if config.codex.testcase:
                return None, generated_codes, test_code, eval_test_pass
            else:
                return None, generated_codes

    # Run the program
    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match, process_guesses_partial)
    except Exception as e:
        if isinstance(e, AssertionError):
            eval_test_pass = 0
        else:
            eval_test_pass = 2
        if config.eval.test_eval and eval_test:
            print(f'Test case failed with GT answer - sample {sample_id}')
            return eval_test_pass
        # print full traceback
        traceback.print_exc()
        if retrying and not config.use_cached_codex2:
            if config.codex.testcase:
                return None, code, test_code, eval_test_pass
            else:
                return None, code

        # if error in GLIP2
        if 'index 0' in str(e) and retrying:
            print(f'Sample {sample_id} failed in VLM stage. {e}')
            if config.codex.testcase:
                return None, code, test_code, eval_test_pass
            else:
                return None, code #return None, generated_codes
        # if another error in GLIP2
        elif 'NoneType' in str(e) and retrying:
            print(f'Sample {sample_id} failed in VLM stage. {e}')
            if config.codex.testcase:
                return None, code, test_code, eval_test_pass
            else:
                return None, code
        else:
            print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
            # Retry again with fixed code
            new_code = "["  # This code will break upon execution, and it will be caught by the except clause
            if config.codex.testcase:
                result = \
                    run_program([new_code, sample_id, image, possible_answers, query, test_code, eval_test, code], queues_in_,
                                input_type_,
                                retrying=True)[0]
            else:
                result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                                     retrying=True)[0]

    if config.eval.test_eval and eval_test:
        return eval_test_pass
    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    if config.codex.testcase:
        if eval_test_pass != 1 and config.e2e_execute:
            generated_codes[0] = codes
        return result, generated_codes, test_code, eval_test_pass
    else:
        return result, generated_codes


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def main():
    from datasets import get_dataset
    if config.eval.eval_only:
        dataset = get_dataset(config.dataset)
        results = pd.read_csv(config.eval.eval_file)
        pred_all = [r for r in results['result']]
        gt_all = [eval(r) for r in results['answer']]
        accuracy = dataset.accuracy(pred_all, gt_all)
        console.print(f'Final accuracy: {accuracy}')


    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets import get_dataset


    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    codex = partial(forward, model_name='codex', queues=[queues_in, queue_results_main])


    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)

    if config.eval.eval_only:
        results = pd.read_csv(config.eval.eval_file)
        pred_all = [r for r in results['result']]
        gt_all = [eval(r) for r in results['answer']]
        accuracy = dataset.accuracy(pred_all, gt_all)
        console.print(f'Final accuracy: {accuracy}')

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()

    if config.codex.testcase:
        with open(config.codex.testcase_prompt) as f:
            testcase_prompt = f.read().strip()

    if config.codex.testcaseGen:
        with open(config.codex.testcaseGen_prompt) as f:
            testcaseGen_prompt = f.read().strip()


    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        codes_all = [r.replace(" -> str","").replace("execute_command(image)","execute_command(image, my_fig, time_wait_between_lines, syntax)") for r in results['code']] #[r for r in results['code']]
        if config.dataset.max_samples is not None:
            codes_all = codes_all[config.dataset.start_sample:config.dataset.start_sample+config.dataset.max_samples]
    if config.use_cached_test_code:
        results = pd.read_csv(config.cached_test_code_path)
        test_codes_all = [r for r in results["test_code"]]
        if config.dataset.max_samples is not None:
            test_codes_all = test_codes_all[config.dataset.start_sample:config.dataset.start_sample+config.dataset.max_samples]
    if config.use_cached_codex2:
        results = pd.read_csv(config.cached_codex2_path)
        codes2_all = [str(r).replace(" -> str","").replace("execute_command(image)","execute_command(image, my_fig, time_wait_between_lines, syntax)") for r in results['code']]
        codes2_all = [str(r).replace("->str", "").replace("execute_command(image)",
                                                          "execute_command(image, my_fig, time_wait_between_lines, syntax)")
                      for r in codes2_all]
        if config.dataset.max_samples is not None:
            codes2_all = codes2_all[config.dataset.start_sample:config.dataset.start_sample+config.dataset.max_samples]

    if not config.execute_code and config.eval.test_eval:
        # Load the results from the file
        results = pd.read_csv(config.cached_codex_path)
        results_cache = [r for r in results['result']]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    if config.load_models.mPLUG_owl:
        from vision_models import ImageCaptionModel
        img_caption_model = ImageCaptionModel(gpu_number=0)

    all_results = []
    all_answers = []
    all_codes = []
    all_ids = []
    all_querys = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    all_test_codes = []

    correct_num = 0
    total_num = 0
    toxic_num = 0

    if config.eval.test_eval:
        corret_all= 0
        wrong_all = 0
        correct_pass =0
        correct_fail = 0
        wrong_pass = 0
        wrong_fail = 0

    if config.save:
        results_dir = pathlib.Path(config['results_dir'])
        results_dir = results_dir / config.dataset.split
        results_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(results_dir.glob('results_*.csv'))


    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # 1) Generate test code
                if config.codex.testcase and not config.use_cached_test_code:
                    test_code = codex(prompt=batch['query'], base_prompt=testcase_prompt, test_case=True)
                elif config.codex.testcase and config.use_cached_test_code:
                    test_code = test_codes_all[i * batch_size:(i + 1) * batch_size]

                # 2) Generate code
                if not config.use_cached_codex:
                    if config.codex.testcaseGen:
                        if not config.use_cached_test_code:
                            assert_code = ['\n'.join(tc[0].split('\n')[2:-1]) for tc in test_code]
                        else:
                            assert_code = ['\n'.join(tc.split('\n')[2:-1]) for tc in test_code]
                        codes = codex(prompt=batch['query'], base_prompt=testcaseGen_prompt, assert_prompt=assert_code)
                    else: # baseline
                        codes = codex(prompt=batch['query'], base_prompt=base_prompt)
                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache
                    if config.use_cached_codex2:
                        codes2 = codes2_all[i * batch_size:(i + 1) * batch_size]
                    elif config.codex.testcaseGen and not config.use_cached_codex2:  # generate code using generted test case
                        assert_code = test_code
                        codes2 = codex(prompt=batch['query'], base_prompt=testcaseGen_prompt, assert_prompt=assert_code)

                # Run the code
                if config.execute_code:
                    if not config.multiprocessing:
                        results = []
                        if config.codex.testcase:
                            if config.eval.wo_VLM and config.use_cached_codex2:
                                for c, sample_id, img, possible_answers, query, tc, parallel_code in \
                                        zip(codes2, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'], test_code, codes):
                                    result = run_program_wo_VLM([c, sample_id, img, possible_answers, query, tc, False], queues_in, input_type)
                                    results.append(result)
                            elif config.use_cached_codex2 or (config.e2e_execute and config.codex.testcaseGen):
                                for c, sample_id, img, possible_answers, query, tc, parallel_code in \
                                        zip(codes2, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'], test_code, codes):
                                    c = c.replace("[PYTHON]", "").replace("PYTHON]", "").replace("[PYTHON", "").replace(
                                        "PYTHON", "").replace("[Instruction]", "")
                                    result = run_program([c, sample_id, img, possible_answers, query, tc, False], queues_in, input_type, codes=parallel_code)
                                    results.append(result)
                            else:
                                for c, sample_id, img, possible_answers, query, tc in \
                                        zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'], test_code):
                                    if config.codex.model in ['Llama-3-8B-Instruct'] and not config.use_cached_codex:
                                        result = run_program([c[0], sample_id, img, possible_answers, query, tc[0], False], queues_in, input_type)
                                    else:
                                        result = run_program([c, sample_id, img, possible_answers, query, tc, False], queues_in, input_type)
                                    results.append(result)
                        else: # Baseline - ViperGPT
                            for c, sample_id, img, possible_answers, query in \
                                    zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                if config.eval.wo_VLM:
                                    result = run_program_wo_VLM([c, sample_id, img, possible_answers, query], queues_in, input_type)
                                elif config.codex.model in ['Llama-3-8B-Instruct'] and not config.use_cached_codex:
                                    result = run_program([c[0], sample_id, img, possible_answers, query], queues_in, input_type)
                                else:
                                    c = c.replace("[PYTHON]", "").replace("PYTHON]", "").replace("[PYTHON","").replace("PYTHON", "").replace("[Instruction]", "")
                                    c = c.replace("ImagePatch(image_path)", "ImagePatch(image)")
                                    c = c.replace("image_path", "image, my_fig, time_wait_between_lines, syntax")
                                    result = run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                                results.append(result)
                    else:
                        if config.codex.testcase:
                            results = list(pool.imap(partial(
                                run_program, queues_in_=queues_in, input_type_=input_type),
                                zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'],
                                    batch['query'], test_code)))
                        else:
                            results = list(pool.imap(partial(
                                run_program, queues_in_=queues_in, input_type_=input_type),
                                zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])))
                else:
                    results = [(None, c) for c in codes]
                    warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                  "'execute_code' to False by default, because executing code generated by a language "
                                  "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                  "it.")


                all_results += [r[0] for r in results]
                all_codes += [r[1][0] for r in results]

                all_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['query_type']
                all_querys += batch['query']
                all_img_paths += [dataset.get_sample_path(idx) for idx in batch['index']]

                if config.codex.testcase:
                    if config.codex.model in ['Llama-3-8B-Instruct']:
                        if config.use_cached_test_code:
                            all_test_codes += [r for r in test_code]
                        else:
                            all_test_codes += [r[0] for r in test_code]
                    else:
                        all_test_codes += [r[0] for r in test_code]

                # evaluate test case
                # result_assert_result and result_t  => 0 : assertion error, 1: pass, 2: compilation error
                if config.eval.test_eval: # result_assert_result and result_t :
                    if not config.execute_code:  # get the results from the file
                        results = results_cache[i * batch_size:(i + 1) * batch_size]

                    for c, sample_id, answer, possible_answers, query, tc, result_assert_result, img in \
                            zip(codes, batch['sample_id'], batch['answer'], batch['possible_answers'],
                                batch['query'], test_code, results, batch['image']):
                        if config.dataset.dataset_name in ['RefCOCO','RefCOCO+']:
                            if config.eval.confusion_matrix:
                                # evaluating test case using the file - only evaluating test case
                                iou = dataset.accuracy([result_assert_result], [answer])
                                if iou[0] > 0.7:  # case where the answer is same as result
                                    corret_all += 1
                                    result_t = run_program([c, sample_id, img, possible_answers, query, tc, True, [result_assert_result] ], queues_in,
                                                       input_type) # result of putting answer to test case
                                    if result_t == 1:
                                        correct_pass += 1
                                    else:
                                        correct_fail += 1
                                else:  # case where the answer is different from result
                                    result_t = run_program([c, sample_id, img, possible_answers, query, tc, True, [result_assert_result]], queues_in,
                                                       input_type)
                                    wrong_all += 1
                                    if result_t == 1:
                                        wrong_pass += 1
                                    else:
                                        wrong_fail += 1

                            else:
                                result_t = run_program([c, sample_id, img, possible_answers, query, tc, True, answer], queues_in,
                                                       input_type)  # result of putting answer to test case
                                correct_num += 1 if result_t == 1 else 0
                                total_num += 1 if result_assert_result[3] != 2 else 0  # w/o runtime error
                                if result_t == 0 and result_assert_result[3] == 1:
                                    toxic_num += 1
                                    print(f'Test case toxic - sample {sample_id}')


                        else:
                            if config.codex.model in ['Llama-3-8B-Instruct'] and not config.use_cached_codex:
                                result_t = run_program([c, sample_id, answer, possible_answers, query, tc[0], True],
                                                       queues_in, input_type)
                            else:
                                # evaluating test case using the file - only evaluating test case
                                if answer == result_assert_result: # case where the answer is same as result
                                    corret_all += 1
                                    result_t = run_program([c, sample_id, answer, possible_answers, query, tc, True], queues_in,
                                                     input_type) # result of putting answer to test case
                                    if result_t == 1:
                                        correct_pass +=1
                                    else:
                                        correct_fail +=1
                                else: # case where the answer is different from result
                                    result_t = run_program([c, sample_id, result_assert_result, possible_answers, query, tc, True], queues_in,
                                                     input_type)
                                    wrong_all += 1
                                    if result_t == 1:
                                        wrong_pass +=1
                                    else:
                                        wrong_fail +=1


                if i % config.log_every == 0:
                    try:
                        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')
                    if config.eval.test_eval and config.execute_code: # Previous test case evaluation
                        console.print(f'Test Case Accuracy at Batch {i}/{n_batches}: {correct_num/len(all_results)}')
                        console.print(f'Test Case Toxic rate at Batch {i}/{n_batches}: {toxic_num /total_num}')
                        print(total_num, correct_num, toxic_num)
                    elif config.eval.test_eval and config.use_cached_codex:
                        if config.dataset.dataset_name in ['RefCOCO', 'RefCOCO+']:
                            console.print(f'Test Case Accuracy at Batch {i}/{n_batches}: {correct_num / len(all_results)}')
                            console.print(f'Test Case Toxic rate at Batch {i}/{n_batches}: {toxic_num / total_num}')
                            print(total_num, correct_num, toxic_num)
                        else:
                            console.print(f'Test Case Accuracy at Batch {i}/{n_batches}: {corret_all/(corret_all+wrong_all)}')
                            console.print(f'Correct Pass: {correct_pass}, Correct Fail: {correct_fail}, Wrong Pass: {wrong_pass}, Wrong Fail: {wrong_fail}')
                            console.print(f'Correct ALL: {corret_all}, Wrong ALL: {wrong_all}')
                    if config.save:
                        if len(existing_files) == 0:
                            filename = 'results_0.csv'
                        else:
                            filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                                str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
                        print('Saving results to', filename, 'at epoch', i)

                        if not config.codex.testcase:
                            df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths,
                                               all_possible_answers]).T
                            df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
                        else:
                            df = pd.DataFrame([all_results, all_answers, all_codes, all_test_codes, all_ids, all_querys,
                                               all_img_paths,
                                               all_possible_answers]).T
                            df.columns = ['result', 'answer', 'code', 'test_code', 'id', 'query', 'img_path',
                                          'possible_answers']
                        # make the result column a string
                        df['result'] = df['result'].apply(str)
                        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.eval.test_eval and config.execute_code:
        console.print(f'Test Case Final accuracy: {correct_num/len(all_results)}')
        console.print(f'Test Case Final Toxic rate : {toxic_num / total_num}')
        print(total_num, correct_num, toxic_num)
    elif config.eval.test_eval and config.use_cached_codex:
        if config.dataset.dataset_name in ['RefCOCO', 'RefCOCO+']:
            console.print(f'Test Case Final accuracy: {correct_num / len(all_results)}')
            console.print(f'Test Case Final Toxic rate : {toxic_num / total_num}')
            print(total_num, correct_num, toxic_num)
        else:
            console.print(f'Test Case Accuracy at Batch {i}/{n_batches}: {corret_all / (corret_all + wrong_all)}')
            console.print(
                f'Correct Pass: {correct_pass}, Correct Fail: {correct_fail}, Wrong Pass: {wrong_pass}, Wrong Fail: {wrong_fail}')
            console.print(f'Correct ALL: {corret_all}, Wrong ALL: {wrong_all}')

    if config.save:
        if not config.execute_code:
            if not config.save_new_results:
                filename = 'results.csv'
            else:
                if len(existing_files) == 0:
                    filename = 'results_0.csv'
                else:
                   filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                     str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        print('Saving results to', filename)
        if not config.codex.testcase:
            df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_querys, all_img_paths,
                               all_possible_answers]).T
            df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
        else:
            df = pd.DataFrame([all_results, all_answers, all_codes, all_test_codes, all_ids, all_querys, all_img_paths,
                               all_possible_answers]).T
            df.columns = ['result', 'answer', 'code', 'test_code', 'id', 'query', 'img_path', 'possible_answers']
        # make the result column a string
        df['result'] = df['result'].apply(str)
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')

        if config.wandb:
            wandb.log({'accuracy': accuracy})
            wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})

    finish_all_consumers()


if __name__ == '__main__':
    main()
