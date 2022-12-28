# SYDE750 Waterworld

## install
```
docker build -t swarm-group-form docker/.
```

## Interesting results:
### Potential field:
#### general approach, no cooperation
![test_results/potential_field/30_sensor_no_coop.gif](test_results/potential_field/30_sensor_no_coop.gif)

#### single sensor, 0.1 radians rotation per frame
![test_results/potential_field/1_sensor_0.1angle_rot.gif](test_results/potential_field/1_sensor_0.1angle_rot.gif)

### DDPG
#### wait strategy
![test_results/ddpg/APEX_DDPG_custom_waterworld_1_80d21_00000_0_2022-03-28_11-18-45/checkpoint_000870/123.76011435254367.gif](test_results/ddpg/APEX_DDPG_custom_waterworld_1_80d21_00000_0_2022-03-28_11-18-45/checkpoint_000870/123.76011435254367.gif)