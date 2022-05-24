import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotting.plotspecs as specs
from general_utils import set_up_dir
from eval_utils import sum_normalize, min_max_normalize, create_df
import io
from xlsx2html import xlsx2html
import matplotlib.patches as patches
from analysis.flip_utils import collect_softmax_outs 
import spacy
import collections
import matplotlib
from plotting.plotspecs import name_map, replace_rules



def proc_str(x, replace_rules):
    for t0, t1 in replace_rules:
        x = x.replace(t0, t1)
    return x

    

def blend(color, alpha, base=[255,255,255]):
    '''
    color should be a 3-element iterable,  elements in [0,255]
    alpha should be a float in [0,1]
    base should be a 3-element iterable, elements in [0,255] (defaults to white)
    '''
    out = [int(round((alpha * color[i]) + ((1 - alpha) * base[i]))) for i in range(3)]

    return out

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def highlight_max(s):
    is_max = s == s.max()
    dummy = ['background-color: {}'.format(rgb2hex(*blend([255, 0, 0], s[is_max].item()))) if v else '' for v in is_max]
    return dummy


def get_color(s):
    return [255,0,0] if s>=0 else [0,0,255]


def highlight(s):    
    func = lambda x: np.float(x) if x is not None else np.nan#()
    s = s.apply(func)
    is_max = s == s.max()
    dummy =  ['background-color: {}'.format(rgb2hex(*blend(get_color(s_), abs(s_)))) if not np.isnan(s_) else '' for s_ in s]
    return dummy


def plot_df(df, res_dir, name):
    f = os.path.join(res_dir,name)

    df.columns = [proc_str(x, replace_rules).replace(' ','\n') for x in df.columns]
    
    rename_index_dict = {x:proc_str(x, replace_rules) for x in list(df.index)}
    
    df = df.rename(index=rename_index_dict)
    
    
    
    df.style.apply(highlight).to_excel(f+'.xlsx', engine='xlsxwriter',  encoding='utf8')    

    
    # must be binary mode
    xlsx_file = open(f+'.xlsx', 'rb') 
    out_file = io.StringIO()
    xlsx2html(xlsx_file, out_file, locale='en')
    out_file.seek(0)
    result_html = out_file.read()

    # Write HTML String to file.html
    with open(f+'.html', "w") as file:
        file.write(result_html)
    return None


def plot_sentences(dfs_all, plot_cases, label_dict=None, plot_dir=None, N=None, normalize='sum'):
    if plot_dir:
        set_up_dir(plot_dir)

    if N is None:
        N = len(dfs_all[plot_cases[0]])

    check_acc = []
    title_dict = {}
    title_dict['tsr'] = 'Human fixations (TSR)'
    title_dict['ez_nr'] = 'E-Z Reader'
    title_dict['fine_bert_flow_11'] = 'BERT* flow 11'
    for j in range(N):
        f, axs = plt.subplots(nrows=len(plot_cases), ncols=1, figsize=(6, len(plot_cases)))
        for i, case in enumerate(plot_cases):

            df_temp = dfs_all[case].iloc[j]

            if case == 'cnn':
                attn_str = 'x_abs'
            elif 'bert_flow' in case:
                attn_str = 'x_flow_' + case.split('_')[-1]
                df_temp.words = dfs_all['tsr'].iloc[j].words
            elif 'ez_' in case and 'cloze' in case:
                attn_str = 'cloze'
            elif 'ez_' in case:
                attn_str = 'TT'
            else:
                attn_str = 'x'

            #  attn_str = 'x_abs' if case == 'cnn' else 'x'

            x_in = df_temp[attn_str]

            if 'ez_' in case:
                x_in = np.nan_to_num(x_in)

            if normalize == 'sum':
                words, x = df_temp.words, sum_normalize(x_in)
            elif normalize == 'min_max':
                words, x = df_temp.words, min_max_normalize(x_in)
            else:
                ValueError('normalization not known, please choose one out of ["sum", "min_max"]')

            if len(np.shape(x)) == 2:
                x = np.mean(x, 0)

            df = create_df(words, x)
            _, ax = _draw_text_saliency_frame(df, text_size=9, figure=(f, axs[i]), margin=0.5)
            ax.set_xlim((0, ax.get_xlim()[1] + 10))
            ax.axis('off')

            if case in title_dict.keys():
                title_str = title_dict[case]
            else:
                title_str = case

            if ('cnn' in case or 'sattn' in case) and label_dict is not None:  # ['cnn', 'self_attn'] :
                ytrue, ypred = label_dict[df_temp.labels], label_dict[df_temp.ypred]
                title_str = '{} ($y_t$: {} - $\hat y$: {})'.format(title_str, ytrue, ypred)
                check_acc.append(ytrue == ypred)

                ax.set_title(title_str, y=0.85, fontsize=10, x=0.6)
            else:
                # ax.set_title(title_str,y = 0.75, fontsize=10)
                ax.set_title(title_str)

        if plot_dir:
            f.savefig(os.path.join(plot_dir, '{}.png'.format(j)), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    print('check_acc', np.sum(check_acc) / len(check_acc))


def _draw_text_saliency_frame(df,
                              title=None,
                              subtitle=None,
                              inverted_y_axis=True,
                              margin=10,
                              text_size=6,
                              figure=None):
    """
    Args:
        df:
            (pandas.DataFrame) Must contain columns
                text_id, word, area_width, area_height, area_left_x,
                area_right_x, area_bottom_y, area_top_y, heat
            where each value for heat is a float between 0 and 1.
    """
    # avoid side-effects and reset any index
    df = df.copy().reset_index()

    # data must only belong to a single text
    if df.text_id.nunique() != 1:
        raise Error("DataFrame must not contain data of more than one text_id.")

    # compute where text will be drawn inside the word-areas
    area_min_y = "area_top_y" if inverted_y_axis else "area_bottom_y"
    df = df.assign(center_x=lambda df: df.area_left_x + df.area_width / 2,
                   center_y=lambda df: df[area_min_y] + df.area_height / 2)

    # generate graphics
    if figure is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figure

    word_boxes = []
    word_texts = []

    for i, row in df.iterrows():
        if inverted_y_axis:
            lower_left_xy = (row.area_left_x, row.area_top_y)
        else:
            lower_left_xy = (row.area_left_x, row.area_bottom_y)

        if row.heat >= 0:
            attn_col = (1, 0, 0, row.heat)
        else:
            attn_col = (0, 0, 1, abs(row.heat))

        bounding_box = patches.Rectangle(lower_left_xy,
                                         row.area_width,
                                         row.area_height,
                                         facecolor=attn_col,
                                         edgecolor="black",
                                         linewidth=.25)

        word_boxes.append(ax.add_patch(bounding_box))

        word_texts.append(ax.text(row.center_x, row.center_y, row.word,
                                  ha="center", va="center", size=text_size))

    # text area
    max_y = np.max([df.area_top_y, df.area_bottom_y])
    min_y = np.min([df.area_top_y, df.area_bottom_y])
    max_x = np.max([df.area_left_x, df.area_right_x])
    min_x = np.min([df.area_left_x, df.area_right_x])

    # canvas area
    start_x = min_x - margin
    end_x = max_x + margin
    start_y = min_y - margin
    end_y = max_y + margin

    # plot settings
    ax.set_xlim(start_x, end_x)
    ax.set_ylim(start_y, end_y)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, loc="right", size=10)
    if inverted_y_axis:
        # https://stackoverflow.com/questions/2051744/reverse-y-axis-in-pyplot
        ax.invert_yaxis()

    return fig, ax


def image_plot(corrs, tokens_mean, labels, bin_labels, file_name, tokens=True):
    cmap = sns.color_palette("rocket", as_cmap=True)
    if len(corrs)>1:
        max0 = np.max(corrs[0])
        max1 = np.max(corrs[1])
        min0 = np.min(corrs[0])
        min1 = np.min(corrs[1])
        vmax = np.max([max0, max1, abs(min0), abs(min1)])
        vmin = -vmax
    else:
        vmax = np.max([np.max(corrs[0]), np.min(corrs[0])])
        vmin = -vmax

    corrs_SR  = corrs[0]
    if tokens:
        tokens_mean_SR = tokens_mean[0]
        fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=len(corrs), figsize=(4*len(corrs), 2.5),
                                       sharex='col', sharey='row', gridspec_kw={'height_ratios': [5, 1]})
                                                                           # 'width_ratios': [10, 1]})

        if len(corrs)==2:
            ax2 = ax1[1]
            ax1 = ax1[0]
            ax4 = ax3[1]
            ax3 = ax3[0]
    else:

        fig, axs = plt.subplots(nrows=1, ncols=len(corrs), figsize=(4*len(corrs), 3),
                                sharex='col', sharey='row')

        if len(corrs) == 1:
            ax1 = axs
        else:
            ax1 = axs[0]
            ax3 = axs[1]

    im = ax1.imshow(corrs_SR, cmap='PRGn', vmin=vmin, vmax=vmax, aspect=.5)
    ax1.set_yticks([i for i in range(corrs_SR.shape[0])])
    ax1.set_yticklabels(labels)

    ax1.set_xticks(np.arange(0, len(bin_labels[0])))

    if isinstance(bin_labels[0], (type({}.keys()), type({}.values()))):
        ax1.set_xticklabels(bin_labels[0], rotation=90)
    else:
        ax1.set_xticklabels(bin_labels[0])

    for i in range(corrs_SR.shape[0]):
        for j in range(corrs_SR.shape[1]):
            if abs(corrs_SR[i, j]) > 0.5:
                ax1.text(j, i, np.around(corrs_SR[i, j], decimals=2),
                                ha="center", va="center", color="w")

            else:
                ax1.text(j, i, np.around(corrs_SR[i, j], decimals=2),
                                ha="center", va="center", color="k")



    if len(corrs)>1:
        corrs_TSR=corrs[1]
        im = ax2.imshow(corrs_TSR, cmap='PRGn', vmin=vmin, vmax=vmax, aspect=.5)
        ax2.set_xticks(np.arange(0, len(bin_labels[1])))
        ax2.set_xticklabels(bin_labels[1])

        for i in range(corrs_TSR.shape[0]):
            for j in range(corrs_TSR.shape[1]):
                if abs(corrs_TSR[i, j]) > 0.5:
                    ax2.text(j, i, np.around(corrs_TSR[i, j], decimals=2),
                                    ha="center", va="center", color="w")

                else:
                    ax2.text(j, i, np.around(corrs_TSR[i, j], decimals=2),
                                    ha="center", va="center", color="k")

        cax=ax2

    else:
        cax=ax1

    axins = inset_axes(cax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=cax.transAxes,
                       borderpad=0,
                       )

    fig.colorbar(im, cax=axins)

    if tokens:
        width = 0.35  # the width of the bars
        x = np.arange(corrs_TSR.shape[1])
        ax3.bar(x - 5 * width / 6, tokens_mean_SR[0], width / 3, label='TSR',
                color=specs.get_color('tsr')[0])
        ax3.bar(x - 3 * width / 6, tokens_mean_SR[1], width / 3, label='BERT',
                color=specs.get_color('bert')[0])
        ax3.bar(x - width / 6, tokens_mean_SR[2], width / 3, label='RoBERTa',
                color=specs.get_color('roberta')[0])
        ax3.bar(x + width / 6, tokens_mean_SR[3], width / 3, label='T5',
                color=specs.get_color('t5')[0])
        ax3.bar(x + 3 * width / 6, tokens_mean_SR[4], width / 3, label='EZ',
                color=specs.get_color('ez_')[0])
        ax3.bar(x + (5 / 6 * width), tokens_mean_SR[5], width / 3, label='BNC',
                color=specs.get_color('bnc_freq')[0])

        if len(tokens_mean)>1:
            tokens_mean_TSR = tokens_mean[1]

            ax4.bar(x - 5 * width / 6, tokens_mean_TSR[0], width / 3, label='TSR',
                    color=specs.get_color('tsr')[0])
            ax4.bar(x - 3 * width / 6, tokens_mean_TSR[1], width / 3, label='BERT',
                    color=specs.get_color('bert')[0])
            ax4.bar(x - width / 6, tokens_mean_TSR[2], width / 3, label='RoBERTa',
                    color=specs.get_color('roberta')[0])
            ax4.bar(x + width / 6, tokens_mean_TSR[3], width / 3, label='T5',
                    color=specs.get_color('t5')[0])
            ax4.bar(x + 3 * width / 6, tokens_mean_TSR[4], width / 3, label='EZ',
                    color=specs.get_color('ez_')[0])
            ax4.bar(x + (5 / 6 * width), tokens_mean_TSR[5], width / 3, label='BNC',
                    color=specs.get_color('bnc_freq')[0])
            ax4.legend(loc='lower left', bbox_to_anchor=(1.25, .75, 0.5, 0.5))

        else:
            ax3.legend(loc='best', bbox_to_anchor=(.85, .1, 0.5, 0.5))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


def line_plot(corrs, labels, bin_labels, file_name, samples=[], loc=None, rotation=90, xlabel=None, dashed=False):

    color_dict = {}
    # color_dict['BERT* (flow 11)'] = 'fine_bert'
    color_dict['BERT'] = 'bert'
    color_dict['EZ'] = 'ez_'
    color_dict['BNC'] = 'bnc_freq'
    color_dict['RoBERTa'] = 'roberta'
    color_dict['T5'] = 't5'

    # _, ax = plt.subplots(nrows=1, ncols=len(corrs), figsize=(6*len(corrs), 4), sharey='row')
    _, ax = plt.subplots(nrows=1, ncols=len(corrs), figsize=(6*len(corrs), 2), sharey='row')

    if len(corrs)==1:
        ax1 = ax
        ax_leg = ax1
    else:
        ax1 = ax[0]
        ax2 = ax[1]
        ax_leg = ax1

    for ilabel, label in enumerate(labels):
        if dashed:
            ax1.plot(np.arange(corrs[0].shape[1]), corrs[0][ilabel],
                        label=label,
                        marker='o',
                        linestyle='dashed',
                        color=specs.get_color(color_dict[label])[0])

        else:
            ax1.scatter(np.arange(corrs[0].shape[1]), corrs[0][ilabel],
                        label=label,
                        marker='o',
                        color=specs.get_color(color_dict[label])[0])

        if len(corrs)>1:
            if dashed:
                ax2.plot(np.arange(corrs[1].shape[1]), corrs[1][ilabel],
                            label=label,
                            marker='o',
                            linestyle='dashed',
                            color=specs.get_color(color_dict[label])[0])

            else:
                ax2.scatter(np.arange(corrs[1].shape[1]), corrs[1][ilabel],
                        label=label,
                        marker='o',
                        color=specs.get_color(color_dict[label])[0])

            ax2.set_xticks(np.arange(corrs[1].shape[1]))
            ax2.set_xticklabels(bin_labels[1], rotation=rotation)
            ax2.set_xlabel(xlabel)

    ax1.set_xticks(np.arange(corrs[0].shape[1]))
    ax1.set_xticklabels(bin_labels[0], rotation=rotation)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('correlation with human fixation')
    # ax1.legend(loc='best', bbox_to_anchor=(.7, 0.5, 0.5, 0.5))
    if loc is None:
        legend = ax_leg.legend(bbox_to_anchor=(0, -.7), loc="lower left", frameon=1)
    else:
        legend = ax_leg.legend(loc=loc, frameon=1, ncol=2)
    frame = legend.get_frame()
    frame.set_edgecolor('black')

    if len(samples)>0:
        ax3 = ax1.twiny()
        l = ax1.get_xlim()
        l2 = ax3.get_xlim()
        f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
        ticks = f(ax1.get_xticks())
        ax3.xaxis.set_major_locator(tck.FixedLocator(ticks))
        ax3.set_xticklabels(samples[0], fontsize=8)

        if len(corrs)>1:
            ax4 = ax2.twiny()
            l = ax2.get_xlim()
            l2 = ax4.get_xlim()
            ticks2 = f(ax2.get_xticks())
            ax4.xaxis.set_major_locator(tck.FixedLocator(ticks2))
            ax4.set_xticklabels(samples[1], fontsize=8)


    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    
    
import random

def rand_style(k):
    c =  "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    ls = '-'
    lw = 2
    return c, ls,lw 

def plot_flipping(df_flips, model_cases, label_dict, titlestr, flip_case, fracs,  out_dir_flip=None, fax=None, legend_map={}, color_func=rand_style, fontsize=None):
    
    if out_dir_flip:
        set_up_dir(out_dir_flip)
                  

    all_flips = df_flips[flip_case]
    if fax is None:
        f,axs = plt.subplots(1,1, figsize=(10,8))
    else:
        f,axs = fax

    print(flip_case)
    for k in model_cases:
        
        v = all_flips[k]
        
        N_samples = len(v['Evolution'])

        c, ls,lw = color_func(k)

        label = k
        label = legend_map[label] if label in legend_map else label

        if flip_case == 'generate':

            #axs.plot(np.mean(v['E'], axis=0), label=k, color=c, linestyle=ls, linewidth=lw)
            EE = [collect_softmax_outs(v['Evolution'][i], label_map={j:j for j in range(len(label_dict))}) for i in range(N_samples)]
            
            axs.plot(fracs, np.mean(EE, axis=0), label=label, color=c, linestyle=ls, linewidth=lw)
           # axs.set_title('logits$_k$')

        elif  flip_case == 'destroy':

            axs.plot(fracs, np.mean(v['M'], axis=0), label=label, color=c, linestyle=ls, linewidth=lw)
           # axs.set_title('($y_0$-$y_p$)$^2$')

        axs.set_title('{}'.format(titlestr)) #, fontsize=16)

    f.suptitle(flip_case, y=0.95) #, fontsize=22)
    # define mask as 
    if  flip_case == 'destroy':
        axs.set_ylabel('$(f(x)-f(x^m))^2$') #, fontsize=18)
    elif  flip_case == 'generate':
        axs.set_ylabel('class probability', fontsize=16)

    axs.set_xlabel('fraction of tokens flipped')#, fontsize=18)
    if out_dir_flip:
        save_file = os.path.join(out_dir_flip, 'flipping_{}.png'.format(flip_case))
        f.savefig(save_file, dpi=200)

    if fax is None:
        plt.show()   
        
    axs.legend()
    
    
    



def proc_flip_pos(v):
    nlp = spacy.load("en_core_web_sm")
    def proc_(words):
        doc = nlp(words)
        W = []
        T = []
        for token in doc:
           # print(words,  token, token.pos_)
            W.append(token.pos_)
            T.append(token)
        return T, W

    
    First_words = []
    for i in range(len(v['Evolution'])):
        #if len(v['Evolution'][i]['sentence'][0]) <30:
        First_words+= get_evolution_sentence(v['Evolution'][i], verbose=False)
    count=0
    PROPS1=[]

    TOKS = []
    for x1 in First_words:
        x1_ = x1.replace(' ', '').strip()
        OUT = proc_(x1_)
        PROPS1 += OUT[1]
        
        TOKS += OUT[0]
    C = collections.Counter(PROPS1)
    
    N_norm = np.sum(list(C.values()))
    print(N_norm)
    C_norm = {k:float(v)/N_norm for k,v in C.items()}
    
    return TOKS, First_words, C, C_norm,  PROPS1



def get_evolution_sentence(x, verbose=True, label_dict ={}):
    words = np.array(x['sentence'][0])
    words_inds = list(range(len(words)))
    
    start_ind = words_inds[0]
    end_ind = words_inds[-1]
    
    e = x['flip_evolution']
    y_true =  x['y_true']
    
    ns = sorted(e.keys())    
    if len(label_dict)>0:
        print(label_dict[y_true].upper(), '\n')
    flipped_first = []
    for i,frac in enumerate(ns):
        e_ = e[frac]
        inds = e_[1]
        flipped_sen = []
        count = 0
        for j, i_ in enumerate(inds):

            if start_ind in i_:
                flag = '[*'
                count+=1
            elif end_ind in i_:
                flag = '*]'
                count+=1
            else:
                flag=''
                        
            str_ = ' '.join(list(words[i_])) +' ' +  flag
            flipped_sen.append(str_)
                
            if (j>=2) and len(flipped_first) == 0 and count==2:
                flipped_first.append(str_)
        try:
            if verbose:
                print('{:0.2f}'.format( frac), ' '.join(flipped_sen)) #, flag)
        except:
            import pdb;pdb.set_trace()

    return flipped_first



def plot_pos_bars(C_list, ref_dict, fax = None, labels=[], prop_subset=[], color_func =None):
    
    if len(prop_subset)==0:
    
        hilf_ref = sorted([(k,v) for k,v in ref_dict.items()], key= lambda x: x[1])[::-1]
    else:
        hilf_ref = sorted([(k,v) for k,v in ref_dict.items() if k in prop_subset], key= lambda x: x[1])[::-1]

        # sort by prop_subset
        hilf_ref = [(k,ref_dict[k]) for k in prop_subset]

    if fax is not None:
        f,axs = fax
    else:
        f, axs = plt.subplots(1,1, dpi=200, figsize=(10,6))
        
    labels1 =  [x[0] for x in hilf_ref] 
    bar_width = 0.6/len(C_list)
    X = np.array(range(len(labels1))) 
    
    for i,c in enumerate(C_list):

        hilf2 = sorted([(k,v) for k,v in c.items()], key= lambda x: x[1])[::-1]
        dict2 = {k:v for k,v in hilf2}
        w1 = [dict2[k] if k in dict2 else 0.  for k in labels1]

        if color_func:
            axs.bar(X+ i*bar_width, w1, label=labels[i], width=bar_width, color = color_func(labels[i])[0])
        else:
            axs.bar(X+ i*bar_width, w1, label=labels[i], width=bar_width)
        
    axs.set_xticks(X +bar_width)
    axs.set_xticklabels(labels1, rotation=45, fontsize=16)
    axs.set_ylabel('POS fraction', fontsize=16)

    
    
def plot_input_reduction_pos_analysis(df_flips_sr, df_flips_tsr, fracs, model_cases, results_dir, flip_case = 'generate'):
    from plotting.plotspecs import get_color, name_map
    import matplotlib as mpl


    f,axs = plt.subplots(3,2, figsize=(6,6.5), sharex=False, sharey=False, 
        gridspec_kw={'height_ratios':[0.65,0.25,0.1], 'width_ratios':[1,1]})


    #model_cases = [k for k in df_flips_sr['generate'].keys() if k not in filter_out][::-1]
    label_dict = {0: 'negative',1: 'neutral', 2: 'positive'}
    plot_flipping(df_flips_sr,model_cases, label_dict, titlestr='', 
                  flip_case=flip_case, fracs=fracs, fax=(f,axs[0,0]), legend_map=name_map,
                  color_func=get_color)



    v_cnn = df_flips_sr['generate']['cnn0.50']
    v_bert = df_flips_sr['generate']['base_bert_flow_11']
    v_tsr = df_flips_sr['generate']['tsr']
    v_ez = df_flips_sr['generate']['ez_nr']
    v_t5 = df_flips_sr['generate']['base_t5_flow_11']
    v_roberta = df_flips_sr['generate']['base_roberta_flow_11']


    _, _, C_cnn, C_cnn_norm, _ = proc_flip_pos(v_cnn)
    _, _, C_bert, C_bert_norm, _ = proc_flip_pos(v_bert)
    _, _, C_tsr, C_tsr_norm, _ = proc_flip_pos(v_tsr)
    _, _, C_ez, C_ez_norm, _ = proc_flip_pos(v_ez)
    _, _, C_t5, C_t5_norm, _ = proc_flip_pos(v_t5)
    _, _, C_roberta, C_roberta_norm, _ = proc_flip_pos(v_roberta)



    C_list = [C_tsr_norm, C_bert_norm, C_t5_norm, C_roberta_norm, C_ez_norm, C_cnn_norm]
    ref_dict = C_bert
    fax = None
    #   labels=['tsr','fine_bert', 'ez_nr', 'cnn']
    labels=['tsr','base_bert','t5', 'roberta', 'ez_nr', 'cnn']

    plot_pos_bars(C_list, ref_dict, (f, axs[1,0]), labels, prop_subset=['NOUN', 'ADJ',  'PROPN'],color_func=get_color)


    ###########################################


    label_dict = {'award': 0, 'education': 1, 'employer': 2, 'founder': 3, 'job_title': 4, 'nationality': 5, 'political_affiliation': 6, 'visited': 7, 'wife': 8}        
    label_dict = {v:k for k,v in label_dict.items()}

    plot_flipping(df_flips_tsr, model_cases, label_dict, titlestr='', 
                  flip_case=flip_case,  fracs=fracs, fax=(f,axs[0,1]),  legend_map=name_map,
                  color_func=get_color)



    v_cnn = df_flips_tsr['generate']['cnn0.50']
    v_bert = df_flips_tsr['generate']['base_bert_flow_11']
    v_tsr = df_flips_tsr['generate']['tsr']
    v_ez = df_flips_tsr['generate']['ez_nr']
    v_t5 = df_flips_tsr['generate']['base_t5_flow_11']
    v_roberta = df_flips_tsr['generate']['base_roberta_flow_11']

    _, _, C_cnn, C_cnn_norm, _ = proc_flip_pos(v_cnn)
    _, _, C_bert, C_bert_norm, _ = proc_flip_pos(v_bert)
    _, _, C_tsr, C_tsr_norm, _ = proc_flip_pos(v_tsr)
    _, _, C_ez, C_ez_norm, _ = proc_flip_pos(v_ez)
    _, _, C_t5, C_t5_norm, _ = proc_flip_pos(v_t5)
    _, _, C_roberta, C_roberta_norm, _ = proc_flip_pos(v_roberta)


    C_list = [C_tsr_norm, C_bert_norm, C_t5_norm, C_roberta_norm, C_ez_norm, C_cnn_norm]
    ref_dict = C_bert
    fax = None
    labels=['tsr','base_bert','t5', 'roberta', 'ez_nr', 'cnn']

    plot_pos_bars(C_list, ref_dict, (f, axs[1,1]), labels, prop_subset=['PROPN', 'PUNCT', 'NOUN'],color_func=get_color)



    # Styling
    axs[0,1].set_ylabel('')
    axs[1,1].set_ylabel('')


    axs[0,1].yaxis.tick_right()
    axs[1,1].yaxis.tick_right()

    f.suptitle('', fontsize=10, y=0.95)

    leg = axs[0,1].legend()
    leg.remove()


    custom_ticks = [0.3,  0.5,  0.7]
    axs[0,0].set_yticks(custom_ticks)
    axs[0,0].set_yticklabels(custom_ticks, fontsize=16)

    custom_ticks = [0.1, 0.3, 0.5]#,  0.6]
    axs[0,1].set_yticks(custom_ticks)
    axs[0,1].set_yticklabels(custom_ticks, fontsize=16)


    custom_ticks = [0.0, 0.5, 1.0]
    axs[0,0].set_xticks(custom_ticks)
    axs[0,0].set_xticklabels(['0', '0.5', '1'],  fontsize=16)

    axs[0,1].set_xticks(custom_ticks)
    axs[0,1].set_xticklabels(['0', '0.5', '1'],  fontsize=16)


    # Common xlabel
    axs[0, 0].set_xlabel('.', color=(0, 0, 0, 0))
    axs[0, 1].set_xlabel('.', color=(0, 0, 0, 0))
    # Make common axis labels
    hilf = axs[0, 0].xaxis.label.get_position()
    f.text(hilf[0], 0.505, 'fraction of tokens flipped', va='center', ha='center', fontsize= 1.8*axs[0, 0].xaxis.label.get_fontsize())

    # Put a legend below current axis
    h1, l1 = axs[0,0].get_legend_handles_labels()    
    h2, l2 = axs[0,1].get_legend_handles_labels()

    axs[2,0].axis('off')
    axs[2,1].axis('off')
    axs[2,0].legend(h1[::-1], l1[::-1], 
                    bbox_to_anchor=(1.,1.6, 0,0), 
                    loc=9, ncol=2, fontsize=14)

    leg = axs[0,0].legend()
    leg.remove()

    custom_ticks = [0.1,0.3]
    axs[1,0].set_yticks(custom_ticks)
    axs[1,0].set_yticklabels(custom_ticks, fontsize=16)
    axs[1,0].set_ylim([0.1, 0.3])


    custom_ticks = [0.1,0.5]
    axs[1,1].set_yticks(custom_ticks)
    axs[1,1].set_yticklabels(custom_ticks, fontsize=16)
    axs[1,1].set_ylim([0.1, 0.5])

    plt.subplots_adjust(hspace=.9, wspace=0.1)

    
    if results_dir:
        set_up_dir(results_dir)
        save_file = os.path.join(results_dir, 'flipping_{}_pos_analysis.pdf'.format(flip_case))
        f.savefig(save_file, dpi=300, bbox_inches = "tight")
    else:
        plt.show()
        

def alpha_blending(hex_color, alpha) :
    """ alpha blending as if on the white background.
    """
    foreground_tuple  = matplotlib.colors.hex2color(hex_color)
    foreground_arr = np.array(foreground_tuple)
    final = tuple( (1. -  alpha) + foreground_arr*alpha )
    return(final)


def plot_errs(ax, x,y ,err, dict_):
    plotline, caplines, barlinecols  = ax.errorbar(x,y, yerr=err, lolims=False, lw=0.8, ls='None', **dict_)
    
    
def plot_ranking(df_in, plot_dir, task, include_rows): 

    from plotting.plotspecs import get_color
    dfs = df_in

    sort_column = 'sentence'
    for k in ['spearman', 'pearson']: #dfs.items():

        fig, ax = plt.subplots(1,1, figsize = (6.2, 3.8))

        width = 0.33
        dx=0.09
        for j,ref_column in enumerate(['sentence', 'token']): #'spearman', 'pearson']): #'pearson']):

            if ref_column==sort_column:
                df_ref = dfs[ref_column]#[['names', 'spearman', 'p_spearman',  'pearson', 'p_pearson' ]]
                df_ref = df_ref[df_ref['names'].isin(include_rows)]

                print(list(df_ref['names']))

                names_ = [proc_str(x, replace_rules) for x in list(df_ref['names'])]
                df_ref['names'] = names_

                df_ref = df_ref.set_index('names')
                print(k)
                df_ref = df_ref.sort_values(by = k)
                
                sort_names= list(df_ref.index)
                labels = list(df_ref.index)
                colors = [ get_color(x)[0] for x in labels]

            else:
                df_ref = dfs[ref_column]
                df_ref = df_ref[df_ref['names'].isin(include_rows)]

                names_ = [proc_str(x, replace_rules) for x in list(df_ref['names'])]
                df_ref['names'] = names_
                df_ref = df_ref.set_index('names')
                
                df_ref = df_ref.loc[sort_names]

            means = df_ref[k]
            print(ref_column,k)
            for a,b in zip(labels, means):
                print('{}:\t{}'.format(a,'{:0.4f}'.format(b)))

            std = df_ref['std_' + k]
            pvals = df_ref['p_' + k]

            ind = np.arange(len(means))  # the x locations for the groups

            labels_cleaned = [x.replace('_abs','').replace('_TT','') for x in labels]
            labels = [name_map[x] if x in name_map else x  for x in labels_cleaned]
          

            plot_pvals = True
            if plot_pvals:

                pval_text = []
                for p_ in pvals:
                    if True:
                        if p_<=0.01:
                            pval_text.append('')
                        elif p_<=0.05:
                            pval_text.append('∗')
                        else:
                            pval_text.append('ns')

                    else:
                        pval_text.append('{:0.2f}'.format(p_))

            hatches = ['///' if 'flow' in l else None for l in labels]
            
            alpha = 0.6
            edge_colors = [alpha_blending(c, alpha+0.1) for c in colors]
            error_kw_dict = dict( capsize=0.0, capthick=0.0, color='black') #, lolims=False)
            
  
            if j==0:
                rects1 = ax.bar(ind+j*width+j*dx , means, width, 
                                label='', hatch=hatches, linewidth=1., color = colors,  edgecolor='#616161', alpha=alpha) 
           
                # plot edge
                ax.bar(ind+j*width+j*dx , means, width,
                                 linewidth=1., color = 'none',  edgecolor='black', alpha=alpha) # color='#9d9d9d'

        
            else:
                rects1 = ax.bar(ind+j*width+j*dx , means, width, 
                                label='', hatch=hatches, linewidth=1., color = colors, edgecolor='#616161', alpha=alpha) 
                
                
                ax.bar(ind+j*width+j*dx , means, width,
                                 linewidth=1., color = 'none',  edgecolor=edge_colors) #, alpha=alpha) # color='#9d9d9d'



            plot_errs(ax, ind+j*width+j*dx, means, std, error_kw_dict)

            if plot_pvals:
                for x_, p_ in zip(ind, pval_text):
                    ax.text(x_+j*width+j*dx, 0.17, p_, fontsize=8, rotation=90, ha='center', va='center')
   
        ax.set_ylabel('correlation to human attention'.format(ref_column), fontsize=12)
        ax.set_xticks(ind+0.25)
        ax.set_xticklabels( labels, rotation=45,  ha="right", rotation_mode="anchor")


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ylim_up = 0.83 if k=='sentence' else 0.69

        ax.set_ylim([-0.07 if task=='SR' else -0.22, ylim_up]) # 1.2*np.max(means)])
        ax.set_title('{}'.format(k), fontsize=14) #, y=-0.15)


        ####
        if task =='SR':
            ypos_list = [1.69]*len(ax.get_xticklabels())
        else:
            ypos_list = [1.35]*len(ax.get_xticklabels())

        new_lim = ax.get_xlim()
        ax.set_xlim([new_lim[0]+0.2, 0.97*new_lim[1]])

        print(k)

        if k=='spearman':
            p0 = matplotlib.patches.Patch(facecolor='#d6d6d6', edgecolor='black')
            p1 = matplotlib.patches.Patch(facecolor='#d6d6d6', edgecolor='#d6d6d6')
            p2 = matplotlib.patches.Patch(facecolor='white', edgecolor='#616161', hatch='///',linewidth=0.1)

            leg = ax.legend([p0,p1, p2], ['sentence', 'token', 'flow'], loc='upper left') 
            for patch in leg.get_patches():
                patch.set_height(8)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05) #, bottom=0.5) 

        if plot_dir:
            set_up_dir(plot_dir)
            fig.savefig(os.path.join(plot_dir, 'system_ranking_{}_{}_{}.png'.format(ref_column,k, task)), dpi=300,  transparent=True)
        plt.show()
        plt.close()

    return dfs
