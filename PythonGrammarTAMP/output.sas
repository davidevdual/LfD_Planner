begin_version
3
end_version
begin_metric
0
end_metric
15
begin_variable
var0
-1
2
Atom grasped(right_hand, pitcher_base)
NegatedAtom grasped(right_hand, pitcher_base)
end_variable
begin_variable
var1
-1
2
Atom grasped(right_hand, mug)
NegatedAtom grasped(right_hand, mug)
end_variable
begin_variable
var2
-1
2
Atom grasped(right_hand, initial_location_lefthand)
NegatedAtom grasped(right_hand, initial_location_lefthand)
end_variable
begin_variable
var3
-1
2
Atom at(right_hand, initial_location_lefthand)
NegatedAtom at(right_hand, initial_location_lefthand)
end_variable
begin_variable
var4
-1
2
Atom at(right_hand, mug)
NegatedAtom at(right_hand, mug)
end_variable
begin_variable
var5
-1
2
Atom at(right_hand, pitcher_base)
NegatedAtom at(right_hand, pitcher_base)
end_variable
begin_variable
var6
-1
2
Atom raised(right_hand, pitcher_base, location_raised)
NegatedAtom raised(right_hand, pitcher_base, location_raised)
end_variable
begin_variable
var7
-1
2
Atom grasped(left_hand, pitcher_base)
NegatedAtom grasped(left_hand, pitcher_base)
end_variable
begin_variable
var8
-1
2
Atom grasped(left_hand, initial_location_righthand)
NegatedAtom grasped(left_hand, initial_location_righthand)
end_variable
begin_variable
var9
-1
2
Atom at(left_hand, mug)
NegatedAtom at(left_hand, mug)
end_variable
begin_variable
var10
-1
2
Atom at(left_hand, initial_location_righthand)
NegatedAtom at(left_hand, initial_location_righthand)
end_variable
begin_variable
var11
-1
2
Atom at(left_hand, pitcher_base)
NegatedAtom at(left_hand, pitcher_base)
end_variable
begin_variable
var12
-1
2
Atom grasped(left_hand, mug)
NegatedAtom grasped(left_hand, mug)
end_variable
begin_variable
var13
-1
2
Atom approached(right_hand, pitcher_base, mug)
NegatedAtom approached(right_hand, pitcher_base, mug)
end_variable
begin_variable
var14
-1
2
Atom poured(right_hand, pitcher_base, mug)
NegatedAtom poured(right_hand, pitcher_base, mug)
end_variable
0
begin_state
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
end_state
begin_goal
2
12 0
14 0
end_goal
26
begin_operator
approach left_hand initial_location_lefthand initial_location_righthand
0
1
0 10 1 0
1
end_operator
begin_operator
approach left_hand initial_location_lefthand mug
0
1
0 9 1 0
1
end_operator
begin_operator
approach left_hand initial_location_lefthand pitcher_base
0
1
0 11 1 0
1
end_operator
begin_operator
approach left_hand initial_location_righthand mug
2
10 0
8 0
1
0 9 1 0
1
end_operator
begin_operator
approach left_hand initial_location_righthand pitcher_base
2
10 0
8 0
1
0 11 1 0
1
end_operator
begin_operator
approach left_hand mug initial_location_righthand
2
9 0
12 0
1
0 10 1 0
1
end_operator
begin_operator
approach left_hand mug pitcher_base
2
9 0
12 0
1
0 11 1 0
1
end_operator
begin_operator
approach left_hand pitcher_base initial_location_righthand
2
11 0
7 0
1
0 10 1 0
1
end_operator
begin_operator
approach left_hand pitcher_base mug
2
11 0
7 0
1
0 9 1 0
1
end_operator
begin_operator
approach right_hand initial_location_lefthand mug
2
3 0
2 0
1
0 4 1 0
1
end_operator
begin_operator
approach right_hand initial_location_lefthand pitcher_base
2
3 0
2 0
1
0 5 1 0
1
end_operator
begin_operator
approach right_hand initial_location_righthand initial_location_lefthand
0
1
0 3 1 0
1
end_operator
begin_operator
approach right_hand initial_location_righthand mug
0
1
0 4 1 0
1
end_operator
begin_operator
approach right_hand initial_location_righthand pitcher_base
0
1
0 5 1 0
1
end_operator
begin_operator
approach right_hand mug initial_location_lefthand
2
4 0
1 0
1
0 3 1 0
1
end_operator
begin_operator
approach right_hand mug pitcher_base
2
4 0
1 0
1
0 5 1 0
1
end_operator
begin_operator
approach right_hand pitcher_base initial_location_lefthand
2
5 0
0 0
1
0 3 1 0
1
end_operator
begin_operator
approach right_hand pitcher_base mug
2
5 0
0 0
2
0 13 -1 0
0 4 1 0
1
end_operator
begin_operator
enclose left_hand initial_location_righthand
1
10 0
1
0 8 -1 0
1
end_operator
begin_operator
enclose left_hand mug
1
9 0
1
0 12 -1 0
1
end_operator
begin_operator
enclose left_hand pitcher_base
1
11 0
1
0 7 -1 0
1
end_operator
begin_operator
enclose right_hand initial_location_lefthand
1
3 0
1
0 2 -1 0
1
end_operator
begin_operator
enclose right_hand mug
1
4 0
1
0 1 -1 0
1
end_operator
begin_operator
enclose right_hand pitcher_base
1
5 0
1
0 0 -1 0
1
end_operator
begin_operator
pour right_hand pitcher_base mug location_raised
2
13 0
6 0
1
0 14 -1 0
1
end_operator
begin_operator
raise right_hand pitcher_base location_raised
1
0 0
1
0 6 -1 0
1
end_operator
0
