
class Config:
    seq_len = 20

    intercept_strategy_choice_num = 10
    intercept_strategy_choice_interval = 2
    intercept_strategy_choice = [i for i in range(0, intercept_strategy_choice_num*intercept_strategy_choice_interval, intercept_strategy_choice_interval)]

    # plm special
    plm_types = ['llama', 'qwen2']  # the types of plm you can use
    plm_sizes = ['small', 'base', 'large', 'xl',]  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_embed_sizes = {
        'qwen2': {
            'small': 896,
        },
        'llama': {
            'base': 4096,
        },
        'qwen2.5': {
            'base': 3584,  # 7B
        },
    }
    plm_layer_sizes = {
        'qwen2': {
            # 'base': 24,
            'small': 24,
            # 'large': 24,
            # 'xl': 24
        },
        'llama': {
            'base': 32,
        },
        'qwen2.5': {
            'base': 28,  # 7B
        },

    }


cfg = Config()
