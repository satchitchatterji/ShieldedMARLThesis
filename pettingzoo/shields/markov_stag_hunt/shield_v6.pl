% actions
action(0)::action(left);
action(1)::action(right);
action(2)::action(up);
action(3)::action(down);
action(4)::action(stay).

sensor_value(0)::sensor(left).
sensor_value(1)::sensor(right).
sensor_value(2)::sensor(up).
sensor_value(3)::sensor(down).
sensor_value(4)::sensor(stag_near_self).
sensor_value(5)::sensor(stag_near_other).

% hunt stag if it is near and there is another agent
% stag_surrounded :- sensor(stag_near_self), sensor(stag_near_other).
hunt :- action(Dir), sensor(Dir).

% run away from stag if it is near and there is no other agent
% stag_alone :- sensor(stag_near_self), \+sensor(stag_near_other).
% hunt_alone :- action(Dir), sensor(Dir), stag_alone.
% waiting is fine if stag is near and there is no other agent
% hunt_alone :- action(stay), stag_alone

safe_next :- \+hunt, sensor(stag_near_self).
safe_next :- \+sensor(stag_near_self).
% safe_next :- hunt_alone.
% safe_next :- \+hunt_alone.
% any action is safe if a stag is not near
% safe_next :- action(_), \+sensor(stag_near_self).
% unsafe_next:- \+hunt.
% safe_next:- \+unsafe_next.
