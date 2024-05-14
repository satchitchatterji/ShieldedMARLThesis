% actions
action(0)::action(left);
action(1)::action(right);
action(2)::action(up);
action(3)::action(down);
action(4)::action(stay).

% 4 sensors indicating the relative position of the stag 
sensor_value(0)::sensor(left).
sensor_value(1)::sensor(right).
sensor_value(2)::sensor(up).
sensor_value(3)::sensor(down).
sensor_value(4)::sensor(stag_near_self).
sensor_value(5)::sensor(stag_near_other).

% hunt stag if it is near and there is another agent
hunt :- action(Dir), sensor(Dir).
% stag_surrounded :- sensor(stag_near_self), sensor(stag_near_other).
% both_hunt :- hunt, stag_surrounded.
% one_hunt :- hunt, sensor(stag_near_self), \+sensor(stag_near_other).
% dont_freeze :- \+action(stay), sensor(stag_near_self), \+stag_surrounded.

safe_next :- \+hunt, sensor(stag_near_self).
safe_next :- action(_), \+sensor(stag_near_self). 
% safe_next :- both_hunt.
% safe_next :- \+one_hunt.
% safe_next :- dont_freeze.
% safe_next :- action(_), \+sensor(stag_near_self), \+stag_surrounded.
% safe_next :- sensor(stag_near_self), \+hunt, \+stag_surrounded.
% safe_next :- \+hunt, sensor(stag_near_self), \+sensor(stag_near_other).