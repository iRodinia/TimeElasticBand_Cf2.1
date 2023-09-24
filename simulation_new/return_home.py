import os
from pycrazyswarm import Crazyswarm

TAKEOFF_DURATION = 5.

def main():
    cf_settings = os.path.abspath(os.path.dirname(__file__)) + "/crazyflies.yaml"
    swarm = Crazyswarm(crazyflies_yaml=cf_settings)
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(5.0)
    cf.goTo([-1.03, -1.19, 0.05], 0., 5.0)
    timeHelper.sleep(5.0)
    cf.land(targetHeight=0.06, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()
