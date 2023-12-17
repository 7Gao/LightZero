from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 8
evaluator_env_num = 1
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(1e3)
reanalyze_ratio = 0
# # ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

smartcross_muzero_config = dict(
    exp_name=f'data_mz_ctree/smartcross_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        config_path='/home/qi/Workspace/github/DI-smartcross/smartcross/envs/cityflow_grid/cityflow_auto_grid_config.json',
        obs_type=['phase'],
        continuous=False,
        from_discrete=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        max_episode_duration=1000,
        green_duration=30,
        yellow_duration=5,
        red_duration=0,
        stop_value=0,
        manager=dict(shared_memory=False, context='spawn', retry_type='renew',),
        muzero=True,
    ),
    policy=dict(
        model=dict(
            observation_shape=24,
            action_space_size=24,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

smartcross_muzero_config = EasyDict(smartcross_muzero_config)
main_config = smartcross_muzero_config

smartcross_muzero_create_config = dict(
    env=dict(
        # Must use the absolute path. All the following "import_names" should obey this too.
        import_names=['smartcross.envs.cityflow_env'],
        type='cityflow_env',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)

smartcross_muzero_create_config = EasyDict(smartcross_muzero_create_config)
create_config = smartcross_muzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        from lzero.entry import train_muzero_with_gym_env as train_muzero
    else:
        raise ValueError(f"Unknown entry_type: {entry_type}")
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
