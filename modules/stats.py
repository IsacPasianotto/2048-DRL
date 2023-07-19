import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import modules.envs as envs
import modules.agents as agents

def play_game(env, random_agent, verbose=False):
    """ Play one game with the given random agent and environment.

    Returns:

    score: the score of the game
    t: the duration of the game
    cumreward: the cumulative reward of the game
    """

    env.reset()
    done = False
    random_agent.reset()
    t = 0

    while not done:
        t += 1
        action = random_agent.act(env.legit_actions)
        _, reward, done, won, _ = env.step(action, verbose=verbose)
        random_agent.cumreward += reward
    return env.score, t, random_agent.cumreward

def play_games(env, random_agent, n_games=10, verbose=False):
    """ Play n_games with the given random_agent and environment.

    Returns:

    a dictionary with the following keys:
        scores: array of scores
        durations: array of durations
        cumrewards: array of cumulative rewards
    """

    scores = np.zeros(n_games)
    durations = np.zeros(n_games)
    cumrewards = np.zeros(n_games)
    for i in range(n_games):
        score, t, cumreward = play_game(env, random_agent, verbose=verbose)
        scores[i] = score
        durations[i] = t
        cumrewards[i] = cumreward

    return {"Scores":scores,
            "Durations": durations,
            "Cumrewards": cumrewards}

def compare_agents(scores_agent):
    """ Compare the scores of the agent with a random agent by making an histogram (it runs n games for the random agent).

    Returns:

    a histogram of the scores of the agent and the random agent (you have to plot it yourself with plt.show()
    """

    env = envs.Game2048Env(log_rewards=True)
    
    random_agent = agents.RandomAgent()
    scores_random_player = play_games(env, random_agent, n_games=len(scores_agent), verbose=False)['Scores']
    hist_agent_scores((scores_agent, scores_random_player), names_agents=["DQN Agent", "Random Agent"], bins=np.floor(np.sqrt(len(scores_agent))).astype(int))


def hist_agent_scores(scores_agents, names_agents, bins=10, alpha=0.3, density=True, title="Histogram of 2048 scores"):
    """ Plot an histogram of the scores of the agents.
    
    Parameters:

    scores_agents: a tuple of arrays of scores of the agents
    names_agents: a tuple of names of the agents
    bins: number of bins of the histogram
    alpha: transparency of the histogram
    density: if True, the integral of the histogram will sum to 1
    title: title of the histogram
    """

    for score_agent, name_agents in zip(scores_agents, names_agents):
        plt.hist(score_agent, bins=bins, alpha=alpha, density=density, label=name_agents)
   
    plt.legend(loc='upper right')
    plt.xlabel('Scores')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ylabel('Frequency')

    title += " for"
    title += " names_agents[0]" + " and" + " names_agents[1]"

    plt.title(title)
    # get prettier histogram
    sns.despine()

def plot_5_metrics(agent, n_data=None, smooth=True, n_ticks=30):
    """ Plot the 5 metrics of the agent cumreward, duration, loss, score and max_tile.

    Returns:

    a plot of the 5 metrics of the agent (you have to plot it yourself with plt.show()
    """

    if n_data is None:
        n_data = len(agent.cumulative_reward)

    cumulative_rewards = agent.cumulative_reward
    max_durations = agent.max_duration
    loss_history = agent.loss_history
    score = agent.score
    max_tile = agent.max_tile
    # create a dataframe out of the rewards
    df = pd.DataFrame(cumulative_rewards, columns=['cumulative_reward'])
    df['max_duration'] = max_durations
    df['loss'] = loss_history
    df['episode'] = df.index
    df['score'] = score
    df['max_tile'] = max_tile
    if smooth:
        df['ewma_cumreward'] = df.cumulative_reward.ewm(span=10).mean()
        df['ewma_score'] = df.score.ewm(span=10).mean()
        df['ewma_loss'] = df.loss.ewm(span=10).mean()
        df['ewma_duration'] = df.max_duration.ewm(span=10).mean()
        df['ewma_max_tile'] = df.max_tile.ewm(span=10).mean()

    df = df.iloc[15:]
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))

    # set integer ticks for x-axis every 30 indices
    for ax in axes:
        ax.set_xticks(df.index[::n_ticks])
        # set values
        ax.set_xticklabels(df.index[::n_ticks])
    
    # change the figure size
    fig.set_size_inches(18.5, 10.5)
    df[:n_data].plot(x='episode', y='cumulative_reward', ax=axes[0])
    df[:n_data].plot(x='episode', y='max_duration', ax=axes[1])
    df[:n_data].plot(x='episode', y='loss', ax=axes[2])
    df[:n_data].plot(x='episode', y='score', ax=axes[3])
    df[:n_data].plot(x='episode', y='max_tile', ax=axes[4])

    # plot in red the ema
    if smooth:
        df[:n_data].plot(x='episode', y='ewma_cumreward', ax=axes[0], color='red')
        df[:n_data].plot(x='episode', y='ewma_duration', ax=axes[1], color='red')
        df[:n_data].plot(x='episode', y='ewma_loss', ax=axes[2], color='red')
        df[:n_data].plot(x='episode', y='ewma_score', ax=axes[3], color='red')
        df[:n_data].plot(x='episode', y='ewma_max_tile', ax=axes[4], color='red')

def stackplot_actions(agent):
    """ Plot the action distribution of the agent over time.

    Returns:

    a plot of the action distribution of the agent over time (you have to plot it yourself with plt.show()
    """

    action_dist = np.array(agent.action_dist)
    # make a stackplot of the action distribution
    plt.stackplot(np.arange(len(action_dist)), action_dist.T, labels=[f"Action {i}" for i in range(4)])
    plt.legend(loc='upper right')
    plt.xlabel("Step")
    plt.ylabel("Action Distribution")
    plt.title("Action Distribution over Time for Agent " + agent.name)

def jointplot_score_duration(scores_agent, durations_agent, name_agent):
    """ Plot the jointplot of the score and duration of the agent.

    Returns:

    a jointplot of the score and duration of the agent (you have to plot it yourself with plt.show()
    """

    # create a dataframe out of the rewards
    df = pd.DataFrame(scores_agent, columns=['score'])
    df['duration'] = durations_agent
    # plot the jointplot
    sns.jointplot(x='duration', y='score', data=df, kind='hex', color='red')
    plt.title("Score vs Duration for Agent " + name_agent)

def heatmap_mean_board(agent):
    """ Plot the heatmap of the mean board of the agent.

    Returns:

    a heatmap of the mean board of the agent (you have to plot it yourself with plt.show()
    """

    mean_board = np.array(agent.mean_board)
    # plot the heatmap
    sns.heatmap(mean_board, cmap='Blues')
    plt.title("Mean Board for Agent " + agent.name)

