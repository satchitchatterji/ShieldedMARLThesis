% actions
action(0)::action(left);
action(1)::action(right);
action(2)::action(up);
action(3)::action(down);
action(4)::action(stay).

sensor_value(0)::sensor(stag_left).
sensor_value(1)::sensor(stag_right).
sensor_value(2)::sensor(stag_up).
sensor_value(3)::sensor(stag_down).
sensor_value(4)::sensor(stag_near_self).
sensor_value(5)::sensor(stag_near_other).

% go to stag if it is near and there is another agent
stag_surrounded :- sensor(stag_near_self), sensor(stag_near_other).
hunt :- action(left), sensor(stag_left), stag_surrounded.
hunt :- action(right), sensor(stag_right), stag_surrounded.
hunt :- action(up), sensor(stag_up), stag_surrounded.
hunt :- action(down), sensor(stag_down), stag_surrounded.

% run away from stag if it is near and there is no other agent
stag_alone :- sensor(stag_near_self), \+sensor(stag_near_other).
run_away :- action(left), sensor(stag_left), stag_alone.
run_away :- action(right), sensor(stag_right), stag_alone.
run_away :- action(up), sensor(stag_up), stag_alone.
run_away :- action(down), sensor(stag_down), stag_alone.
run_away :- action(stay), stag_alone.

safe_next :- hunt.
safe_next :- run_away. 
% any action is safe if a stag is not near
safe_next:- action(_), \+sensor(stag_near_self).
% unsafe_next:- \+hunt.
% safe_next:- \+unsafe_next.
