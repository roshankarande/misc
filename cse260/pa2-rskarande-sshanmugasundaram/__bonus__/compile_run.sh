# # sudo nvidia-smi -ac 5001,1590
# make -C build_T4 clean
# make -C build_T4 `cat src_todo_T4/OPTIONS.txt`
# ./gen_results.sh | tee curr_results.txt

# make -C build_T4 cublastest=1 `cat src_todo_T4/OPTIONS.txt`
# ./gen_results.sh | tee results/bench/curr_results_bench.txt


if [[ $1 -eq 0 ]]; then
     make -C build_T4/ clean
fi

if [[ $1 -eq 1 ]]; then
    # sudo nvidia-smi -ac 5001,1590
    make -C build_T4 clean
    make -C build_T4 `cat src_todo_T4/OPTIONS.txt`
    ./gen_results.sh | tee "results/output/$(date +"%d_%I_%M").txt"
fi

if [[ $# -eq 0 ]]; then
    ./gen_results.sh | tee "results/output/$(date +"%d_%I_%M").txt"
fi