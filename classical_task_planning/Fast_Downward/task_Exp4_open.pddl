;; problem file: task_Exp4_open.pddl
;; This problem file is to run Fast Downward for the "open" task. It is for Experiment 4.

(define (problem bimanual-probOpen)
  (:domain domain_2)
(:objects bleach_cleanser cracker_box gelatin_box marker master_chef_can mustard_bottle pitcher_base potted_meat_can sugar_box tomato_soup_can tuna_fish_can initial_location_handleft initial_location_handright - graspable 
            handright handleft - hand
            1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 - location 
  )

  (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
  )
 (:goal (and (opened cracker_box 6)))) 