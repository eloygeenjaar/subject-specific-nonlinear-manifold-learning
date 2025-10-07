import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

input_sizes = [1000, 10000, 100000]
hidden_sizes = [64, 128, 256, 512]
num_subjects = [16, 32, 64, 128, 256, 512, 1024, 2048]

fig, axs = plt.subplots(len(input_sizes), len(hidden_sizes), figsize=(15, 10))
for (i_s, input_size) in enumerate(input_sizes):
    for (h_s, hidden_size) in enumerate(hidden_sizes):
        # Weight + bias
        g_ls = [input_size * hidden_size + hidden_size] * len(num_subjects)
        s_ls, d_ls = [], []
        for ns in num_subjects:
            # Subject weight + bias
            s_ls.append(
                ns * input_size * hidden_size + ns * hidden_size
            )
            # Decomposed weight + bias
            d_ls.append(
                (ns * hidden_size) +  # Subject-specific weights
                (input_size * hidden_size) + # U/VT
                (hidden_size * hidden_size) + # U/VT
                (hidden_size)
            )

        axs[i_s, h_s].set_title(f'IS: {input_size}, HS: {hidden_size}')
        axs[i_s, h_s].plot(num_subjects, d_ls, linewidth=3, alpha=0.9, color='#8776B3')
        axs[i_s, h_s].plot(num_subjects, g_ls, linewidth=3, alpha=0.9, color='#B8E3FF')
        axs[i_s, h_s].plot(num_subjects, s_ls, linewidth=3, alpha=0.9, color='#C8B0EE')
        axs[-1, h_s].set_xlabel('Number of subjects')
        axs[i_s, 0].set_ylabel('Number of parameters (log)')
        axs[i_s, h_s].set_yscale('log')


axs[i_s, h_s].legend(['Decomposed', 'Group', 'Subject'])
plt.tight_layout()
plt.savefig('figures/figure6.png', dpi=400)