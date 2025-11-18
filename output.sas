begin_version
3
end_version
begin_metric
0
end_metric
6
begin_variable
var0
-1
2
Atom at(handright, wood_block)
NegatedAtom at(handright, wood_block)
end_variable
begin_variable
var1
-1
2
Atom enclosed(handright, wood_block)
NegatedAtom enclosed(handright, wood_block)
end_variable
begin_variable
var2
-1
2
Atom at(handleft, wood_block)
NegatedAtom at(handleft, wood_block)
end_variable
begin_variable
var3
-1
2
Atom enclosed(handleft, wood_block)
NegatedAtom enclosed(handleft, wood_block)
end_variable
begin_variable
var4
-1
2
Atom at(experimenter_hand, wood_block)
NegatedAtom at(experimenter_hand, wood_block)
end_variable
begin_variable
var5
-1
2
Atom passed(wood_block, 15, experimenter_hand, 7)
NegatedAtom passed(wood_block, 15, experimenter_hand, 7)
end_variable
0
begin_state
1
1
1
1
1
1
end_state
begin_goal
1
5 0
end_goal
7
begin_operator
approach experimenter_hand wood_block
0
1
0 4 1 0
1
end_operator
begin_operator
approach handleft wood_block
0
1
0 2 1 0
1
end_operator
begin_operator
approach handright wood_block
0
1
0 0 1 0
1
end_operator
begin_operator
enclose handleft wood_block
1
2 0
1
0 3 1 0
1
end_operator
begin_operator
enclose handright wood_block
1
0 0
1
0 1 1 0
1
end_operator
begin_operator
pass handleft handright wood_block 15 experimenter_hand 7
3
4 1
3 0
1 0
1
0 5 -1 0
1
end_operator
begin_operator
pass handright handleft wood_block 15 experimenter_hand 7
3
4 1
3 0
1 0
1
0 5 -1 0
1
end_operator
0
