@rem plot rewards (see in -h)
python result_analysis/plot_reward.py log-xxx.txt

@rem plot accuracy (see in -h)
python result_analysis/plot_accuracy.py log-xxx.txt log-yyy.txt

@rem extract drop number file:
python result_analysis/extract_drop_num.py log-xxx.txt -o yyy.txt
