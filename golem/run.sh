# Ground truth: 20-node ER2 graph
# Data: Linear DAG model with Gaussian-NV noise
python src/main.py  --seed 1 \
                    --d 20 \
                    --graph_type ER \
                    --degree 4 \
                    --noise_type gaussian_nv \
                    --equal_variances \
                    --lambda_1 2e-2 \
                    --lambda_2 5.0 \
                    --checkpoint_iter 5000