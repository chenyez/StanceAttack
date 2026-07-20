# nohup bash ./run_pstance_train_feb26.sh > run_pstance_train_feb26_result.log 2>&1 &

dataset_name=covid
step1
test_data=/home/czhao43/adv_stance/datasets/covid19/raw_test_all_onecol2.csv
tosave_path=./annotation_results/rephrased_results_train_sacarsm_chatgpt_cot_testonly2_gemini_step1_covid.csv
python iterative_eval_chatgpt_mistral_cot2_victim_GEMINI_step1.py -dataset_name ${dataset_name} -model_in_use Bart_encoder -ex_random_seed 0 -model chatgpt -max_tries 40 -test_data ${test_data} -tosave_path ${tosave_path}



# # #step2 bart encoder
test_data=/home/czhao43/adv_stance/datasets/covid19/raw_test_all_onecol2.csv
cot_step1_result=./annotation_results/rephrased_results_train_sacarsm_chatgpt_cot_testonly2_gemini_step1_covid.csv
tosave_path=./annotation_results/rephrased_results_train_sacarsm_chatgpt_cot_testonly2_gemini_step2_covid.csv
python iterative_eval_chatgpt_mistral_cot2_victim_GEMINI_step2.py -dataset_name ${dataset_name} -model_in_use Bart_encoder -cot_step1_result ${cot_step1_result} -ex_random_seed 0 -model chatgpt -max_tries 100 -test_data ${test_data} -tosave_path ${tosave_path}





