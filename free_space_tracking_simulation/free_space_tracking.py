"""
Main Simulation Script

Run as:

    $ python3 getting_started.py --overrides ./parameters.yaml

"""
import time
import inspect
import numpy as np
import pybullet as p
from rich.tree import Tree
from rich import print

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger
from simulation.controlling.PIDcontroller import PIDController, thrusts
from free_space_planner import Free_Space_Planner

def run():

    # Start a timer.
    START = time.time()

    # Load configuration.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    config.quadrotor_config['ctrl_freq'] = 60
    config.quadrotor_config['pyb_freq'] = 240
    config.quadrotor_config['gui'] = True

    CTRL_FREQ = config.quadrotor_config['ctrl_freq']
    CTRL_DT = 1/CTRL_FREQ

    env = make('quadrotor', **config.quadrotor_config)
    obs, info = env.reset()
    controller = PIDController()
    
    # Create a logger and counters
    logger = Logger(logging_freq_hz=CTRL_FREQ)
    episodes_count = 1
    cumulative_cost = 0
    collisions_count = 0
    collided_objects = set()
    violations_count = 0
    episode_start_iter = 0
    time_label_id = p.addUserDebugText("", textPosition=[0, 0, 1],physicsClientId=env.PYB_CLIENT)
    stats = []

    # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}
    commander = Free_Space_Planner(obs, info, verbose=config.verbose, logger=logger)

    # Initial printouts.
    if config.verbose:
        print('\tInitial observation [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0]: ' + str(obs))
        print('\tControl timestep: ' + str(info['ctrl_timestep']))
        print('\tControl frequency: ' + str(info['ctrl_freq']))
        print('\tMaximum episode duration: ' + str(info['episode_len_sec']))
        print('\tNominal quadrotor mass and inertia: ' + str(info['nominal_physical_parameters']))
        
        print('\tFinal target hover position [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]: ' + str(info['x_reference']))
        print('\tDistribution of the error on the initial state: ' + str(info['initial_state_randomization']))
        print('\tDistribution of the error on the inertial properties: ' + str(info['inertial_prop_randomization']))
        print('\tDistribution of the error on positions of obstacles: ' + str(info['obs_randomization']))
        print('\tDistribution of the disturbances: ' + str(info['disturbances']))
        
        print('\tA priori symbolic model:')
        print('\t\tState: ' + str(info['symbolic_model'].x_sym).strip('vertcat'))
        print('\t\tInput: ' + str(info['symbolic_model'].u_sym).strip('vertcat'))
        print('\t\tDynamics: ' + str(info['symbolic_model'].x_dot).strip('vertcat'))
        print('Input constraints lower bounds: ' + str(env.constraints.input_constraints[0].lower_bounds))
        print('Input constraints upper bounds: ' + str(env.constraints.input_constraints[0].upper_bounds))
        print('State constraints active dimensions: ' + str(config.quadrotor_config.constraints[1].active_dims))
        print('State constraints lower bounds: ' + str(env.constraints.state_constraints[0].lower_bounds))
        print('State constraints upper bounds: ' + str(env.constraints.state_constraints[0].upper_bounds))
        print('\tSymbolic constraints: ')
        for fun in info['symbolic_constraints']:
            print('\t' + str(inspect.getsource(fun)).strip('\n'))
    
    reference_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        radius=0.03,
                                        rgbaColor=[0.,1,0.,1],
                                        physicsClientId=env.PYB_CLIENT)


    ep_start = time.time()
    first_ep_iteration = True
    for i in range(config.num_episodes*CTRL_FREQ*env.EPISODE_LEN_SEC):

        # episodes_count == 1, resampler takes effect

        # Elapsed sim time.
        curr_time = (i-episode_start_iter)*CTRL_DT

        # Print episode time in seconds on the GUI.
        time_label_id = p.addUserDebugText("Ep. time: {:.2f}s".format(curr_time),
                                           textPosition=[0, 0, 1.5],
                                           textColorRGB=[1, 0, 0],
                                           lifeTime=3*CTRL_DT,
                                           textSize=1.5,
                                           parentObjectUniqueId=0,
                                           parentLinkIndex=-1,
                                           replaceItemUniqueId=time_label_id,
                                           physicsClientId=env.PYB_CLIENT)

        # Compute control input.
        if first_ep_iteration:
            done = False
            info = {}
            first_ep_iteration = False
        
        if episodes_count == 1:
            target_pos, target_vel, target_acc, target_omega, target_yaw = commander.cmdSimulation(curr_time, obs, resample=True, info=info)
        elif episodes_count == 2:
            target_pos, target_vel, target_acc, target_omega, target_yaw = commander.cmdSimulation(curr_time, obs, resample=False, info=info)
        else:
            target_pos, target_vel, target_acc, target_omega, target_yaw = commander.cmdSimulation(curr_time, obs, info=info)
        action = thrusts(controller, commander.CTRL_TIMESTEP, commander.KF, obs, target_pos, target_vel)
        obs, cost, done, info = env.step(action)

        # Show the reference position
        p.createMultiBody(baseMass=0,
                        baseVisualShapeIndex=reference_shape,
                        basePosition=target_pos,
                        useMaximalCoordinates=True,
                        physicsClientId=env.PYB_CLIENT)
        

        # Update the controller internal state and models.
        commander.onlineAnalysis(curr_time, action, obs, cost, done, info)

        # Add up cost, collisions, violations.
        cumulative_cost += cost
        if info["collision"][1]:
            collisions_count += 1
            collided_objects.add(info["collision"][0])
        if 'constraint_values' in info and info['constraint_violation'] == True:
            violations_count += 1

        # Printouts.
        if config.verbose and i%int(CTRL_FREQ/2) == 0:
            print('\n'+str(i)+'-th step.')
            print('\tApplied action: ' + str(action))
            print('\tObservation: ' + str(obs))
            print('\tCost: ' + str(cost) + ' (Cumulative: ' + str(cumulative_cost) +')')
            print('\tDone: ' + str(done))
            print('\tAt goal position: ' + str(info['at_goal_position']))
            print('\tTask completed: ' + str(info['task_completed']))
            if 'constraint_values' in info:
                print('\tConstraints evaluations: ' + str(info['constraint_values']))
                print('\tConstraints violation: ' + str(bool(info['constraint_violation'])))
            print('\tCollision: ' + str(info["collision"]))
            print('\tTotal collisions: ' + str(collisions_count))
            print('\tCollided objects (history): ' + str(collided_objects))

        # Log data.
        pos = [obs[0],obs[2],obs[4]]
        rpy = [obs[6],obs[7],obs[8]]
        vel = [obs[1],obs[3],obs[5]]
        bf_rates = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=(i-episode_start_iter)/CTRL_FREQ,
                   state=np.hstack([pos, np.zeros(4), rpy, vel, bf_rates, np.sqrt(action/env.KF)]),
                   target=np.hstack([target_pos, target_vel, target_acc, target_omega, target_yaw]),
                   control=action
                   )

        # Synchronize the GUI.
        if config.quadrotor_config.gui:
            sync(i-episode_start_iter, ep_start, CTRL_DT)

        # If an episode is complete, reset the environment.
        if done:
            # Analysis and save episode results
            commander.EpisodeAnalysis()

            # Plot logging (comment as desired).
            logger.plot(comment="get_start-episode-"+str(episodes_count), autoclose=True)

            # Save logged data as npz package
            if episodes_count == 1:
                logger.save(new_feature=True)
            else:
                logger.save(new_feature=False)

            # Append episode stats.
            if config.quadrotor_config.done_on_collision and info["collision"][1]:
                termination = 'COLLISION'
            elif config.quadrotor_config.done_on_completion and info['task_completed']:
                termination = 'TASK COMPLETION'
            elif config.quadrotor_config.done_on_violation and info['constraint_violation']:
                termination = 'CONSTRAINT VIOLATION'
            else:
                termination = 'MAX EPISODE DURATION'
            if commander.interstep_learning_occurrences != 0:
                interstep_learning_avg = commander.interstep_learning_time/commander.interstep_learning_occurrences
            else:
                interstep_learning_avg = commander.interstep_learning_time
            episode_stats = [
                '[yellow]Flight time (s): '+str(curr_time),
                '[yellow]Reason for termination: '+termination,
                '[green]Total cost: '+str(cumulative_cost),
                '[red]Number of collisions: '+str(collisions_count),
                '[red]Number of constraint violations: '+str(violations_count),
                '[white]Total and average interstep learning time (s): '+str(commander.interstep_learning_time)+', '+str(interstep_learning_avg),
                '[white]Interepisode learning time (s): '+str(commander.interepisode_learning_time),
                ]
            stats.append(episode_stats)

            # Create a new logger.
            logger = Logger(logging_freq_hz=CTRL_FREQ)
            commander.logger = logger

            # Reset/update counters.
            episodes_count += 1
            if episodes_count > config.num_episodes:
                break
            cumulative_cost = 0
            collisions_count = 0
            collided_objects = set()
            violations_count = 0

            # Reset the environment.
            new_initial_obs, _ = env.reset()
            first_ep_iteration = True

            commander.interEpisodeReset()

            if config.verbose:
                print(str(episodes_count)+'-th reset.')
                print('Reset obs' + str(new_initial_obs))
            
            episode_start_iter = i+1
            ep_start = time.time()

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    print(str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} sec, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
          .format(i,
                  env.CTRL_FREQ,
                  config.num_episodes,
                  elapsed_sec,
                  i/elapsed_sec,
                  (i*CTRL_DT)/elapsed_sec
                  )
          ))

    # Print episodes summary.
    tree = Tree("Summary")
    for idx, ep in enumerate(stats):
        ep_tree = tree.add('Episode ' + str(idx+1))
        for val in ep:
            ep_tree.add(val)
    print('\n\n')
    print(tree)
    print('\n\n')


if __name__ == "__main__":
    run()