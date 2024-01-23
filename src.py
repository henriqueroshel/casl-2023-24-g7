import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from scipy import stats
from collections import namedtuple, defaultdict
from tqdm import tqdm   # progress bar

seed = 171048

# movement vectors on the hexgrid (using axial coordinates)
DIRECTION_VECTORS = np.array([ (-1,0), (-1,+1), (0,-1), (0,+1), (+1,0), (+1,-1) ])
def available_directions(position, map_size):
    # filter the available directions of movement of a individual given its position
    q,r = position
    s = -q-r
    possible_dirs = DIRECTION_VECTORS
    # avoid individual of going beyond map edges
    if abs(q) == map_size-1:
        possible_dirs = possible_dirs[ possible_dirs[:,0] != q/abs(q) ]
    if abs(r) == map_size-1:
        possible_dirs = possible_dirs[ possible_dirs[:,1] != r/abs(r) ]
    if abs(s) == map_size-1:
        possible_dirs = possible_dirs[ possible_dirs.sum(axis=1) != -s/abs(s) ]
    return possible_dirs

# the initial population of each of the S species is placed respectively
# on the following rq-coordinates on the hex map
rng1 = np.random.default_rng(seed)
def initial_population_position(S, size):
    size_ = size-1
    # order of initial coordinates for species to be placed (away from each other)
    if S == 1: # center of the map
        return [(0,0)]
    elif S == 3: # equidistant
        return [ (-size_,0), (0,+size_), (+size_,-size_) ]
    else: # hexagon vertices and center
        order = [ (+size_,-size_), (-size_,+size_), (-size_,0), (+size_,0), (0,+size_), (0,-size_), (0,0) ]
    if S > 7: # more than 7 species -> randomly allocated in different tiles
        while len(order) < S:
            q = rng1.integers(-size_,+size_)
            r_lim = size_-abs(q)
            r = 0 if r_lim==0 else rng1.integers(-abs(r_lim),+abs(r_lim)+1)
            if (q,r) not in order:
                order.append((q,r))
        return order
    return order

class Event:
    def __init__(self, event_handler, time, fes, individuals_list, params_dict):
        self.time = time
        self.fes = fes
        self.individuals_list = individuals_list
        # the function and its parameters of the event
        self.event_handler = event_handler 
        self.params_dict = params_dict
        
    def execute(self):
        self.event_handler(clock=self.time, fes=self.fes, individuals_list=self.individuals_list, **self.params_dict)

class FutureEventSet:
    # implementation of the Future Event Set as a priority queue
    def __init__(self):
        self.items = []
    def __len__(self):
        return len(self.items)
    def is_empty(self):
        return len(self) == 0
        
    def put(self, event):
        self.items.append(event)
    def pop(self):
        # pop next event (lowest time) if there is events on the FES
        if self.is_empty():
            print("FutureEventSet is empty")
            return   
        next_event = min(self.items, key=lambda ev:ev.time)
        self.items.remove( next_event )
        return next_event
    def update_death_time(self, individual, new_death_time):
        # get death event of individual
        for i,ev in enumerate(self.items):
            if ev.event_handler==death and ev.params_dict['individual']==individual:
                death_event_index = i
                break
        # update death event time 
        self.items[death_event_index].time = new_death_time

rng2 = np.random.default_rng(seed)
class Individual(object):
    # model for the individual
    def __init__(self, species, parent=None, lifetime=None, position=None):
        # parent is None when individual belongs to generation 0
        # lifetime and position are None when individual does not belong to generation 0
        self._id = Individual.id_counter()
        self.parent = parent
        self.lifetime = lifetime
        self.position = position
        self.generation = parent.generation+1 if parent else 0
        self.species = species

    def __hash__(self):
        return self._id
    def __repr__(self):
        return f'Individual({self._id})'
    def reset_id_count():
        # called at the beginning of each simulation
        Individual.individual_id = 0
    def id_counter():
        # generates unique id for each individual
        Individual.individual_id += 1
        return Individual.individual_id

    def generate_LF(self, p_improve, alpha):
        # inverse transform method for the given distribution
        lf_parent = self.parent.lifetime
        u = rng2.uniform(0,1)
        if u < (1-p_improve):
            life = lf_parent * (u/(1-p_improve))
        else:
            life = lf_parent * (1 + alpha*(u+p_improve-1)/p_improve)
        return life
    
    def birth(self, birth_time):
        # handle birth of individual
        self.birth_time = birth_time
        self.is_dead = False
        # generation 0 is instantiated with its lifetime and position
        if self.lifetime is None: 
            self.lifetime = self.generate_LF(self.species.p_improve, self.species.alpha)
        if self.position is None: 
            self.position = np.array( self.parent.position )
        # set death time and register lifetime for the species
        self.death_time = self.birth_time + self.lifetime
        self.species.individuals_lifetime.append( self.lifetime )

    def next_child(self, clock):
        # generate next child of individual as a poisson process,
        # i.e. time between births is exponentially distributed
        child_birth_time = clock + rng2.exponential(1 / self.species.reprod_rate)
        child = Individual(self.species, parent=self)
        return child, child_birth_time
    
    def n_moves(self):
        # generate number of moves on next time unit
        return rng2.integers(0, self.species.speed + 1)

    def fight(self, clock, survival_chance):
        # generate penalty to the remaining lifetime
        u_survival = rng2.uniform(0,1)
        if u_survival > survival_chance:
            # individual dies instantly
            penalty = 1
        else:
            # individuals with lower survival chance have a higher penalty
            u_penalty = rng2.uniform(0,1)
            if survival_chance>=0.5:
                penalty = u_penalty * (1-survival_chance)
            else:
                penalty = 1 - u_penalty*survival_chance
        
        # update lifetime and death time
        remaining_lifetime = self.death_time - clock
        new_remaining_lifetime = remaining_lifetime * (1-penalty)
        index = self.species.individuals_lifetime.index( self.lifetime )
        self.lifetime = new_remaining_lifetime + clock - self.birth_time
        self.species.individuals_lifetime[index] = self.lifetime
        self.death_time = clock + new_remaining_lifetime

rng3 = np.random.default_rng(seed)
class Species(object):
    # class for handling a species and storing its characteristics
    def __init__(self, fes, individuals_list, initial_position, initial_population_size, 
                 initial_lifetime_bound, reprod_rate, p_improve, alpha, 
                 speed, strength):
        self._id = Species.id_counter()
        self.initial_population_size = initial_population_size
        self.population_size = 0
        self.population_log = dict() # time:population size
        self.reprod_rate = reprod_rate
        self.p_improve = p_improve
        self.alpha = alpha
        self.speed = speed
        self.strength = strength

        self.is_extinct = False
        self.extinction_time = None
        self.individuals_lifetime = [] # for computing average lifetime

        # we consider the distribution of the remaining lifetime of the initial 
        # population to be uniformly distributed between initial_lifetime_bound
        upper_bound, lower_bound = initial_lifetime_bound
        initial_population_LF = rng3.uniform(upper_bound, lower_bound, size=initial_population_size)
        for i in range(initial_population_size):
            # instantiate each individual of the initial population
            individual = Individual(species=self, lifetime=initial_population_LF[i], position=np.array(initial_position))
            event = Event( birth, 0, fes, individuals_list, params_dict={'individual':individual} )
            fes.put( event )
    def __repr__(self):
        return f'Species({self._id})'
    def reset_id_count():
        # called at the beginning of each simulation
        Species.species_id = 0
    def id_counter():
        # map IDs to "A", "B", "C" ...
        Species.species_id += 1
        return chr(64 + Species.species_id) 

    def get_speed_and_strenght(self):
        return self.speed, self.strength
    def average_lifetime(self):
        return np.mean(self.individuals_lifetime)

    def exponential_growth(self):
        # return True if population grows exponentially;
        # exponential growth is considered when population size is
        # 100 times bigger than initial population size
        return self.population_size > 100*self.initial_population_size
    def stop_simulation(self):
        # return True if population is extinct or grows exponentially
        return (self.is_extinct) or (self.exponential_growth())

    def plot_population(self, ax):
        # plot population size evolution
        time_log = np.fromiter( self.population_log.keys(), dtype='f' )
        size_log = np.fromiter( self.population_log.values(), dtype='i' )
        ax.step(time_log, size_log, where='post', label=self._id, linewidth=1)

def birth(clock, individual, fes, individuals_list):
    # handles the birth of a individual
    species = individual.species
    parent = individual.parent
    if individual.generation>0 and parent.is_dead:
        # ignore children born after parent's death
        return

    # schedule next child of individual's parent
    if individual.parent:
        brother, brother_birth_time = individual.parent.next_child(clock)
        event = Event( birth, brother_birth_time, fes, individuals_list, params_dict={'individual':brother} )
        fes.put( event )

    if species.exponential_growth():
        # ignore children born when population size configures exponential growth
        return

    # birth event for individual
    individual.birth(clock)
    individuals_list.append(individual)
    # updates log of population
    species.population_size += 1
    species.population_log[clock] = species.population_size 

    # schedule first child of individual
    child, child_birth_time = individual.next_child(clock)
    event = Event( birth, child_birth_time, fes, individuals_list, params_dict={'individual':child} )
    fes.put( event )
    # schedule individual's death
    event = Event( death, individual.death_time, fes, individuals_list, params_dict={'individual':individual} )
    fes.put( event )

def death(clock, individual, fes, individuals_list):
    # handles death event of individual
    individual.is_dead = True
    individuals_list.remove(individual)
    # updates log of population
    species = individual.species
    species.population_size -= 1
    species.population_log[clock] = species.population_size
    if species.population_size == 0:
        species.is_extinct = True
        species.extinction_time = clock

rng4 = np.random.default_rng(seed)
def movement_scheduler(clock, map_size, fes, individuals_list):
    # at the beginning of each time unit, we schedule the movement of each individual;
    movement_log = defaultdict(list) # records (time: [tuples (individual, new_position)])
    for individual in individuals_list:
        # each individual perform n_moves uniformly distributed for each time unit
        pos = np.copy(individual.position)
        n_moves = individual.n_moves()

        for m in range(n_moves):
            # choose one of the available directions and record the move
            move_time = clock+(m+1)/n_moves
            directions = available_directions(pos, map_size)
            move = directions[ rng4.integers(0,len(directions)) ]
            pos += move
            movement_log[move_time].append( (individual, pos) )

    for time in movement_log.keys():
        # schedule the actual movement event
        moves_list = movement_log[time]
        move_event = Event(movement, time, fes, individuals_list, params_dict={'moves_list':moves_list})
        fes.put( move_event )

    # schedule next scheduler of movements after one time unit
    schedule_movements_event = Event(movement_scheduler, clock+1, fes, individuals_list, params_dict={'map_size':map_size} )
    fes.put( schedule_movements_event )

def movement(clock, moves_list, fes, individuals_list):
    # perform the moves listed on moves_list
    for individual, position in moves_list:
        individual.position = position
    
    # check if there are encounters of different species
    global_map = defaultdict(set)
    for individual in individuals_list:
        pos = tuple(individual.position)
        global_map[pos].add(individual)
    for pos in global_map:
        individuals_on_pos = global_map[pos]
        species_on_pos = set( ind.species for ind in individuals_on_pos )
        if len(species_on_pos) > 1:
            encounter(clock, individuals_on_pos, fes, individuals_list)

FIGHT_WEIGHTS = [1, 1, 2] # respectively (speed, strength, number of individuals)
def encounter(clock, individuals_on_pos, fes, individuals_list):
    species_on_pos = set(ind.species for ind in individuals_on_pos)
    # randomly select pairs of species to fight
    while len(species_on_pos) >= 2:
        species1 = species_on_pos.pop()
        species2 = species_on_pos.pop()
        individuals1 = set(ind for ind in individuals_on_pos if ind.species==species1)
        individuals2 = set(ind for ind in individuals_on_pos if ind.species==species2)

        # computes the score of the fight based on the 3 criteria and its weights
        speed1, strength1 = species1.get_speed_and_strenght()
        speed2, strength2 = species2.get_speed_and_strenght()
        n1, n2 = len(individuals1), len(individuals2)
        diffs = [ speed1-speed2, strength1-strength2, n1-n2 ]
        # score positive=species1 advantage ; negative=species2 advantage
        score = np.dot(diffs, FIGHT_WEIGHTS)

        # map the score to a value between 0 and 1
        p_survival2 = 1/( 1+np.exp(score) )
        p_survival1 = 1 - p_survival2
        # update the remaining life of each individual involved
        for ind1 in individuals1:
            ind1.fight(clock, p_survival1)
            fes.update_death_time(ind1, ind1.death_time)
        for ind2 in individuals2:
            ind2.fight(clock, p_survival2)
            fes.update_death_time(ind2, ind2.death_time)

def natural_selection_simulator(species_data, map_size, max_simulation_time, verbose=False, plot_animation=False, frame_rate=None):
    """ Perform one simulation for the given input parameters
    Args:
        species_data: list of dicts containing species parameters
        map_size (int): size of the hexagon grid
        max_simulation_time (float): stop condition
        verbose (bool): whether to print simulation state, input and output on stout 
        plot_animation (bool): whether to produce plot and animation of simulation
    Returns:
        species_list: final state of species populations
    """    
    S = len(species_data)
    # get the initial position of each Species;
    # individuals of same species are initialized on the same tile
    initial_position_list = initial_population_position(S, map_size)
    species_list = [None] * S
    Individual.reset_id_count()
    Species.reset_id_count()
    
    fes = FutureEventSet()
    individuals_list = []

    for i in range(S):
        # instantiate each species and schedule birth events of initial individuals
        species_parameters = species_data[i]
        species_position = np.array( initial_position_list[i] )
        species_list[i] = Species( 
            fes=fes, 
            individuals_list=individuals_list, 
            initial_position=species_position, 
            **species_parameters
        )

    # first movement scheduler event
    schedule_movements_event = Event(movement_scheduler, 0., fes, individuals_list, params_dict={'map_size':map_size} )
    fes.put( schedule_movements_event )

    if verbose:
        print(f'+++ INPUT PARAMETERS +++\n- Map size: {map_size}')
        print(f'- Max simulation time: {max_simulation_time:.0f}')
        # use df to better display species' parameters
        species_df = pd.DataFrame(species_data).transpose()
        species_df.columns = pd.Series([sp._id for sp in species_list])
        print(f'- Species: \n{species_df}')
        print('\n+++ SIMULATION +++')
        
    if plot_animation:
        # initializes the animation
        fig,ax = plt.subplots()
        ax.axis('off')
        hex_grid(ax, map_size)
        ax.set_xlim(-map_size*1.5, map_size*1.5)
        ax.set_ylim(-map_size*3**.5, map_size*3**.5)
        artists = []
        frame_rate = max_simulation_time/50 if not frame_rate else frame_rate
        plot_event = Event(map_animation, 0, fes, individuals_list, 
                           params_dict={'ax':ax,'species_list':species_list,'map_size':map_size, 'artists':artists, 'frame_rate':frame_rate})
        fes.put( plot_event )

    # MAIN LOOP
    clock=0
    stop_condition = False
    while not stop_condition:
        # get next event, update clock and perform the event
        event = fes.pop()
        clock = event.time
        event.execute()
        # update stop condition
        stop_condition  = clock>max_simulation_time 
        stop_condition |= all(sp.stop_simulation() for sp in species_list)

        if verbose:
            print(f'\rClock: {clock:.3f} - Population size: {dict((sp._id,sp.population_size) for sp in species_list)}', end=' ', flush=True)
    if verbose:
        print(f'\rClock: {clock:.3f} - Population size: {dict((sp._id,sp.population_size) for sp in species_list)}', flush=True)
        print('Average lifetime:', end=' ', flush=True)
        for sp in species_list:
            print(f'({sp._id} - {sp.average_lifetime():.3f})', end=' ', flush=True)
        print('\nExtinction time:', end=' ', flush=True)
        for sp in species_list:
            if sp.extinction_time:
                print(f'({sp._id} - {sp.extinction_time:.3f})', end=' ', flush=True)

    if plot_animation:
        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles=handles[:S], labels=labels[:S], title='Species', fontsize=9, title_fontsize=9, loc='outside left center')
        for lh in leg.legend_handles: 
            lh.set_alpha(1)
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200, repeat=True, repeat_delay=1500)
        ani.save('animation.gif', writer=animation.PillowWriter(fps=5))
        fig2,ax2 = plot_populations(species_list)
        plt.show()

    return species_list

def hex_grid(ax, map_size):
    # generate the map grid plot
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    for i in range(-map_size+1,+map_size):
        ax.axline((0,i*3**.5), slope=(3**-.5), c='k', alpha=.1, linewidth=1)
        ax.axline((0,i*3**.5), slope=-(3**-.5), c='k',alpha=.1, linewidth=1)
        ax.axvline(x=i*1.5, c='k',alpha=.1, linewidth=1)
def map_animation(ax, artists, clock, fes, individuals_list, species_list, map_size, frame_rate):
    # produces a frame of the animation
    markers = ['o', '^', 's', 'p', 'X', 'd', 'h', 'P', '*']
    artists_ = [ax.annotate(f'time: {clock:.2f}', (0,1), xycoords='axes fraction')]
    for i, sp in enumerate(species_list):
        q_coordinates = np.fromiter((ind.position[0] for ind in individuals_list if ind.species==sp), dtype='i')
        r_coordinates = np.fromiter((ind.position[1] for ind in individuals_list if ind.species==sp), dtype='i')
        # convert to cartesian coordinates
        x = 1.5 * q_coordinates
        y = 3**.5 * -(r_coordinates+q_coordinates/2)
        species_pos = ax.scatter(x,y, marker=markers[i%len(markers)], color=f'C{i}', alpha=.25, label=sp._id, linewidth=.5)
        artists_.append(species_pos)

    artists.append( artists_ )
    # schedule next frame
    plot_event = Event(map_animation, clock+1/frame_rate, fes, individuals_list, 
                       params_dict={'ax':ax, 'species_list':species_list, 'map_size':map_size, 'artists':artists, 'frame_rate':frame_rate})
    fes.put( plot_event )
def plot_populations(species_list,):
    # evolution of populations size for a single simulation
    fig, ax = plt.subplots()
    for species in species_list:
        species.plot_population(ax)
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Population size')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='best', title='Species', title_fontsize=9)
    return fig,ax

def confidence_interval(values, conf_level=0.98):
    # computes the confidence interval of a particular 
    # measure where values is a list of empirical values
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    
    if n<30: # t distribution
        ci_lower,ci_upper = stats.t.interval(conf_level, df=n-1, loc=avg, scale=std/n**.5)
    else: # normal distribution
        ci_lower,ci_upper = stats.norm.interval(conf_level, loc=avg, scale=std/n**.5)
    delta = (ci_upper-ci_lower) / 2
    return avg, delta
def accuracy(value, delta):
    # computes the accuracy of a measure given its value
    # and the semi-width (delta) of the confidence interval    
    eps = delta / value # relative error
    acc = 1 - eps # accuracy
    return max(acc, 0) # return only non-negative values

def n_simulations(species_data, map_size, max_simulation_time, N_sims, conf_level=0.98, verbose=True):
    # perform N_sims simulations and computes the output measures
    S = len(species_data)
    
    if verbose:
        print(f'+++ INPUT PARAMETERS +++\n- Map size: {map_size}')
        print(f'- Max simulation time: {max_simulation_time:.0f}')
        # use df to better display species' parameters
        species_df = pd.DataFrame(species_data).transpose()
        species_df.columns = pd.Series([chr(65+i) for i in range(len(species_data))])
        print(f'- Species: \n{species_df}\n')

    # record output of every simulation
    avg_lifetime_list = np.zeros((N_sims,S))
    extinct_list = np.zeros((N_sims, S))
    print('+++ SIMULATIONS +++')
    for i in tqdm(range(N_sims), ncols=120):
        # perform the N_sims simulations
        species_i = natural_selection_simulator(species_data, map_size, max_simulation_time)
        output_i = np.array([(sp.average_lifetime(), int(sp.is_extinct)) for sp in species_i])

        avg_lifetime_list[i, :] = output_i[:,0]
        extinct_list[i, :] = output_i[:,1]

    # confidence interval and accuracy computation
    lifetime_ci = np.zeros((S,2))
    lifetime_acc = np.zeros(S)
    extinction_prob_ci = np.zeros((S,2))
    extinction_prob_acc = np.zeros(S)
    for i in range(S):
        avg_lifetime, delta_avg_lifetime = confidence_interval(avg_lifetime_list[:, i], conf_level)
        lifetime_ci[i] = [avg_lifetime, delta_avg_lifetime]
        lifetime_acc[i] = accuracy(avg_lifetime, delta_avg_lifetime)

        if len(np.unique(extinct_list[:, i])) > 1:
            extinction_prob, delta_extinct = confidence_interval(extinct_list[:, i], conf_level)
            extinction_prob_ci[i] = [extinction_prob, delta_extinct]
            extinction_prob_acc[i] = accuracy(extinction_prob, delta_extinct)        
        else: # when a species is extinct in every run or in no run
            extinction_prob_ci[i] = [extinct_list[0, i], 0]
            extinction_prob_acc[i] = 1

    print('\n+++ OUTPUT MEASURES +++')
    print('- Average lifetime:')
    for j in range(S):
        avg, delta = lifetime_ci[j]
        print(f'\t - {species_i[j]}: {avg:.3f} \u00b1 {delta:.3f} (accuracy: {lifetime_acc[j]:.3f})')
    print('- Extinction probability:')
    for j in range(S):
        avg, delta = extinction_prob_ci[j]
        print(f'\t - {species_i[j]}: {avg:.3f} \u00b1 {delta:.3f} (accuracy: {extinction_prob_acc[j]:.3f})')


if __name__=='__main__':

    max_simulation_time = 100
    map_size = 10
    columns = ['initial_population_size', 'initial_lifetime_bound', 'reprod_rate', 
               'p_improve', 'alpha', 'speed', 'strength']
    
    # input with 6 similar species
    # data = [
    #     [5, (5, 10), .25, .15, 0.50, 4, 1.32],
    #     [5, (5, 10), .25, .15, 0.50, 3, 3.57],
    #     [5, (5, 10), .25, .15, 0.50, 4, 2.48],
    #     [5, (5, 10), .25, .15, 0.50, 4, 4.82],
    #     [5, (5, 10), .25, .15, 0.50, 4, 2.85],
    #     [5, (5, 10), .25, .15, 0.50, 6, 3.50],
    # ]
    # input with 3 different species
    data = [
        [5, (6, 8), .25, .30, 0.50, 5, 2],
        [5, (8, 10), .25, .20, 0.50, 4, 3],
        [5, (10, 12), .25, .10, 0.50, 6, 4],
    ]    

    S = len(data)
    # convert list of dicts with species parameters
    species_data = []
    for i in range(S):
        species_params = dict()
        for j in range(len(columns)):
            species_params[ columns[j] ] = data[i][j]
        species_data.append(species_params)
    
    print("\nPART 1: single run and validation")
    species_list = natural_selection_simulator(species_data, map_size, max_simulation_time, 
                                               verbose=True, plot_animation=True)

    N_sims = 20         # REDUCED NUMBER OF RUNS FOR SAVING TIME
    N_sims = 1000     # more runs -> results added to the report
    print(f"\n\nPART 2: confidence intervals - {N_sims} simulations")
    n_simulations(species_data, map_size, max_simulation_time, 
                  N_sims, conf_level=0.98, verbose=False)