from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dill
from py_crms.model0 import fit_model_0
from py_crms.model1 import fit_model_1

n_data = 2
# k=j=i=0

def fit_models(k):
    lambda_bias_model_0 = np.zeros((1, 100, 7))
    lambda_bias_model_0[:] = np.nan
    acc_model_0_prop = np.zeros((1, 100, 7))
    acc_model_0_prop[:] = np.nan
    fpr_model_0_prop = np.zeros((1, 100, 7))
    fpr_model_0_prop[:] = np.nan
    fnr_model_0_prop = np.zeros((1, 100, 7))
    fnr_model_0_prop[:] = np.nan

    lambda_bias_model_1 = np.zeros((1, 100, 7))
    lambda_bias_model_1[:] = np.nan
    acc_model_1_prop = np.zeros((1, 100, 7))
    acc_model_1_prop[:] = np.nan
    fpr_model_1_prop = np.zeros((1, 100, 7))
    fpr_model_1_prop[:] = np.nan
    fnr_model_1_prop = np.zeros((1, 100, 7))
    fnr_model_1_prop[:] = np.nan

    for i in range(100):
        for j in range(7):
            file_path = f"Compare/sim_data_i_{i+1}_k_{k+1}_j_{j+1}.csv"
            df = pd.read_csv(file_path)
            sim_data = df.filter(regex="^[^n|^lambda]", axis=1)
            lambda_true = df["lambda"].iloc[0:2]
            results0 = fit_model_0(data=sim_data,
                                   classifier="LR",
                                   true_covid_name="true_c",
                                   lambda_true=lambda_true,
                                   n_data=2)
            lambda_bias_model_0[0, i, j] = results0['bias'][1]
            acc_model_0_prop[0, i, j] = results0['acc'][1]
            fpr_model_0_prop[0, i, j] = results0['fpr'][1]
            fnr_model_0_prop[0, i, j] = results0['fnr'][1]
            results1 = fit_model_1(data=sim_data,
                                   true_covid_name="true_c",
                                   n_data=2,
                                   n_iter=500,
                                   burn_in=100,
                                   lambda_true=lambda_true,
                                   pooled=True)
            lambda_bias_model_1[0, i, j] = results1['bias'][1]
            acc_model_1_prop[0, i, j] = results1['acc'][1]
            fpr_model_1_prop[0, i, j] = results1['fpr'][1]
            fnr_model_1_prop[0, i, j] = results1['fnr'][1]

            print(f"Iteration:  Job = {k},  i = {i}, j = {j} \n")

    results = {"model0": [lambda_bias_model_0,
                          acc_model_0_prop,
                          fpr_model_0_prop,
                          fnr_model_0_prop],
               "model1": [lambda_bias_model_1,
                          acc_model_1_prop,
                          fpr_model_1_prop,
                          fnr_model_1_prop]}

    return results

results = Parallel(n_jobs=5)(delayed(fit_models)(i) for i in range(5))
dill.dump_session('./compare.pkl')
# dill.load_session('./compare.pkl')
len(results)

# FIGURES
coefs_cs = range(-3, 4)
#k = 0
col_names = [f'violation = {k}' for k in coefs_cs]
sns.set_theme()
for k in range(5):
    fig, axs = plt.subplots(figsize=(10, 10), nrows=4)
    fig.subplots_adjust(hspace=0.5)
    # bias
    lambda_bias_model_1 = pd.DataFrame(results[k]['model1'][0][0], columns=col_names)
    lambda_bias_model_0 = pd.DataFrame(results[k]['model0'][0][0], columns=col_names)
    lambda_bias_model_1['fit_model'] = 'model_1'
    lambda_bias_model_0['fit_model'] = 'model_0'
    lambda_bias = lambda_bias_model_1.append(lambda_bias_model_0, ignore_index=True)
    lambda_bias_model = pd.melt(lambda_bias, id_vars='fit_model', value_vars=col_names,
                                value_name='bias', var_name='condition')
    g1a = sns.boxplot(x="condition", y="bias", hue="fit_model",
                      hue_order=["model_0", "model_1"], palette=["r", "b"],
                      data=lambda_bias_model,
                      ax=axs[0])
    g1a.set_title("Bias in the estimated prevalence.")
    g1a.set_xlabel("")
    #g1a.set_ylim(-0.15, 0.15)
    #plt.close()

    # variance

    # acc
    acc_model_1 = pd.DataFrame(results[k]['model1'][1][0], columns=col_names)
    acc_model_0 = pd.DataFrame(results[k]['model0'][1][0], columns=col_names)
    acc_model_1['fit_model'] = 'model_1'
    acc_model_0['fit_model'] = 'model_0'
    acc = acc_model_1.append(acc_model_0, ignore_index=True)
    acc_model = pd.melt(acc, id_vars='fit_model', value_vars=col_names,
                        value_name='accuracy', var_name='condition')
    acc_model
    sns.set_theme()
    g3a = sns.boxplot(x="condition", y="accuracy", hue="fit_model",
                      hue_order=["model_0", "model_1"], palette=["r", "b"],
                      data=acc_model,
                      ax=axs[1])
    g3a.set_title("Accuracy in individual death classification.")
    g3a.set_xlabel("")
    #g3a.set_ylim(0.4, 1.1)
    #plt.close()

    # fpr
    fpr_model_1 = pd.DataFrame(results[k]['model1'][2][0], columns=col_names)
    fpr_model_0 = pd.DataFrame(results[k]['model0'][2][0], columns=col_names)
    fpr_model_1['fit_model'] = 'model_1'
    fpr_model_0['fit_model'] = 'model_0'
    fpr = fpr_model_1.append(fpr_model_0, ignore_index=True)
    fpr_model = pd.melt(fpr, id_vars='fit_model', value_vars=col_names,
                        value_name='fpr', var_name='condition')
    fpr_model
    g4a = sns.boxplot(x="condition", y="fpr", hue="fit_model",
                      hue_order=["model_0", "model_1"], palette=["r", "b"],
                      data=fpr_model,
                      ax=axs[2])
    g4a.set_title("False Positive Rate (FPR) in individual death classification.")
    g4a.set_xlabel("")
    #g4a.set_ylim(0.4, 1.1)
    #plt.close()

    # fnr
    fnr_model_1  = pd.DataFrame(results[k]['model1'][3][0], columns=col_names)
    fnr_model_0 = pd.DataFrame(results[k]['model0'][3][0], columns=col_names)
    fnr_model_1['fit_model'] = 'model_1'
    fnr_model_0['fit_model'] = 'model_0'
    fnr = fnr_model_1.append(fnr_model_0, ignore_index=True)
    fnr_model = pd.melt(fnr, id_vars='fit_model', value_vars=col_names,
                        value_name='fnr', var_name='condition')
    fnr_model
    g5a = sns.boxplot(x="condition", y="fnr", hue="fit_model",
                      hue_order=["model_0", "model_1"], palette=["r", "b"],
                      data=fnr_model,
                      ax=axs[3])
    g5a.set_title("False Negative Rate (FNR) in individual death classification.")
    g5a.set_xlabel("")
    #g5a.set_ylim(-0.1, 1.1)
    #plt.close()

    plt_name = f'Compare/Figures/sim_run_0_box_plot_{k}.png'
    plt.savefig(plt_name)
    plt.close()

# bias
col_names = [f'violation = {k}' for k in coefs_cs]
unknown_values = [0.1, 0.3, 0.5, 0.7, 0.9]
unknown_array = np.repeat([f'unknown = {k}' for k in unknown_values], 100)
model_1_bias_array = np.concatenate([results[0]['model1'][0][0],
                                     results[1]['model1'][0][0],
                                     results[2]['model1'][0][0],
                                     results[3]['model1'][0][0],
                                     results[4]['model1'][0][0]])
model_1_bias_variation = pd.DataFrame(abs(model_1_bias_array), columns=col_names)
model_1_bias_variation['unknown'] = unknown_array
model_1_bias_variation['fit_model'] = 'model_1'

model_0_bias_array = np.concatenate([results[0]['model0'][0][0],
                                     results[1]['model0'][0][0],
                                     results[2]['model0'][0][0],
                                     results[3]['model0'][0][0],
                                     results[4]['model0'][0][0]])
model_0_bias_variation = pd.DataFrame(abs(model_0_bias_array), columns=col_names)
model_0_bias_variation['unknown'] = unknown_array
model_0_bias_variation['fit_model'] = 'model_0'

bias_array = model_1_bias_variation.append(model_0_bias_variation, ignore_index=True)

fig, axs = plt.subplots(figsize=(10, 20), nrows=7)
fig.subplots_adjust(hspace=0.5)
for i in range(7):
    gx = sns.boxplot(x="unknown", y=col_names[i], hue="fit_model",
                     hue_order=["model_0", "model_1"], palette=["r", "b"],
                     data=bias_array,
                     ax=axs[i])
    gx.set_title("Absolute difference in the estimated prevalence with selection bias: " + col_names[i])
    gx.set_xlabel("")
    gx.set_ylabel("bias")
plt_name = f'Compare/Figures/sim_run_0_box_plot_bias.png'
plt.savefig(plt_name)
plt.close()

# acc
col_names = [f'violation = {k}' for k in coefs_cs]
unknown_values = [0.1, 0.3, 0.5, 0.7, 0.9]
unknown_array = np.repeat([f'unknown = {k}' for k in unknown_values], 100)
model_1_acc_array = np.concatenate([results[0]['model1'][1][0],
                                    results[1]['model1'][1][0],
                                    results[2]['model1'][1][0],
                                    results[3]['model1'][1][0],
                                    results[4]['model1'][1][0]])
model_1_acc_variation = pd.DataFrame(model_1_acc_array, columns=col_names)
model_1_acc_variation['unknown'] = unknown_array
model_1_acc_variation['fit_model'] = 'model_1'

model_0_acc_array = np.concatenate([results[0]['model0'][1][0],
                                    results[1]['model0'][1][0],
                                    results[2]['model0'][1][0],
                                    results[3]['model0'][1][0],
                                    results[4]['model0'][1][0]])
model_0_acc_variation = pd.DataFrame(model_0_acc_array, columns=col_names)
model_0_acc_variation['unknown'] = unknown_array
model_0_acc_variation['fit_model'] = 'model_0'

acc_array = model_1_acc_variation.append(model_0_acc_variation, ignore_index=True)

fig, axs = plt.subplots(figsize=(10, 20), nrows=7)
fig.subplots_adjust(hspace=0.5)
for i in range(7):
    gx = sns.boxplot(x="unknown", y=col_names[i], hue="fit_model",
                     hue_order=["model_0", "model_1"], palette=["r", "b"],
                     data=acc_array,
                     ax=axs[i])
    gx.set_title("Accuracy in individual death classification with selection bias: " + col_names[i])
    gx.set_xlabel("")
    gx.set_ylabel("acc")
    gx.legend(loc="center right", bbox_to_anchor=(1.12, .5), fontsize=8)
plt_name = f'Compare/Figures/sim_run_0_box_plot_acc.png'
plt.savefig(plt_name)
plt.close()

# fpr
col_names = [f'violation = {k}' for k in coefs_cs]
unknown_values = [0.1, 0.3, 0.5, 0.7, 0.9]
unknown_array = np.repeat([f'unknown = {k}' for k in unknown_values], 100)
model_1_fpr_array = np.concatenate([results[0]['model1'][2][0],
                                    results[1]['model1'][2][0],
                                    results[2]['model1'][2][0],
                                    results[3]['model1'][2][0],
                                    results[4]['model1'][2][0]])
model_1_fpr_variation = pd.DataFrame(model_1_fpr_array, columns=col_names)
model_1_fpr_variation['unknown'] = unknown_array
model_1_fpr_variation['fit_model'] = 'model_1'

model_0_fpr_array = np.concatenate([results[0]['model0'][2][0],
                                    results[1]['model0'][2][0],
                                    results[2]['model0'][2][0],
                                    results[3]['model0'][2][0],
                                    results[4]['model0'][2][0]])
model_0_fpr_variation = pd.DataFrame(model_0_fpr_array, columns=col_names)
model_0_fpr_variation['unknown'] = unknown_array
model_0_fpr_variation['fit_model'] = 'model_0'

fpr_array = model_1_fpr_variation.append(model_0_fpr_variation, ignore_index=True)

fig, axs = plt.subplots(figsize=(10, 20), nrows=7)
fig.subplots_adjust(hspace=0.5)
for i in range(7):
    gx = sns.boxplot(x="unknown", y=col_names[i], hue="fit_model",
                     hue_order=["model_0", "model_1"], palette=["r", "b"],
                     data=fpr_array,
                     ax=axs[i])
    gx.set_title("False Positive Rate (FPR) in individual death classification with selection bias: " + col_names[i])
    gx.set_xlabel("")
    gx.set_ylabel("fpr")
    gx.legend(loc="center right", bbox_to_anchor=(1.12, .5), fontsize=8)
plt_name = f'Compare/Figures/sim_run_0_box_plot_fpr.png'
plt.savefig(plt_name)
plt.close()

# fnr
col_names = [f'violation = {k}' for k in coefs_cs]
unknown_values = [0.1, 0.3, 0.5, 0.7, 0.9]
unknown_array = np.repeat([f'unknown = {k}' for k in unknown_values], 100)
model_1_fnr_array = np.concatenate([results[0]['model1'][3][0],
                                    results[1]['model1'][3][0],
                                    results[2]['model1'][3][0],
                                    results[3]['model1'][3][0],
                                    results[4]['model1'][3][0]])
model_1_fnr_variation = pd.DataFrame(model_1_fnr_array, columns=col_names)
model_1_fnr_variation['unknown'] = unknown_array
model_1_fnr_variation['fit_model'] = 'model_1'

model_0_fnr_array = np.concatenate([results[0]['model0'][3][0],
                                    results[1]['model0'][3][0],
                                    results[2]['model0'][3][0],
                                    results[3]['model0'][3][0],
                                    results[4]['model0'][3][0]])
model_0_fnr_variation = pd.DataFrame(model_0_fnr_array, columns=col_names)
model_0_fnr_variation['unknown'] = unknown_array
model_0_fnr_variation['fit_model'] = 'model_0'

fnr_array = model_1_fnr_variation.append(model_0_fnr_variation, ignore_index=True)

fig, axs = plt.subplots(figsize=(10, 20), nrows=7)
fig.subplots_adjust(hspace=0.5)
for i in range(7):
    gx = sns.boxplot(x="unknown", y=col_names[i], hue="fit_model",
                     hue_order=["model_0", "model_1"], palette=["r", "b"],
                     data=fnr_array,
                     ax=axs[i])
    gx.set_title("False Negative Rate (FNR) in individual death classification with selection bias: " + col_names[i])
    gx.set_xlabel("")
    gx.set_ylabel("fnr")
    gx.legend(loc="center right", bbox_to_anchor=(1.12, .5), fontsize=8)
plt_name = f'Compare/Figures/sim_run_0_box_plot_fnr.png'
plt.savefig(plt_name)
plt.close()