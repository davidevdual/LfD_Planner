;; problem file: task_Exp4_pass.pddl
;; This problem file is to run Fast Downward for the "pass" task. It is for Experiment 4.
(define (problem bimanual-probPass)
  (:domain domain_pass)
(:objects banana bowl extra_large_clamp master_chef_can mustard_bottle power_drill tomato_soup_can wood_block initial_location_handleft initial_location_handright initial_location_experimenter_hand - graspable 
            handright handleft - hand
            experimenter_hand - experimenterHand
            1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 - location 
  )
    (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
    (at experimenter_hand initial_location_experimenter_hand)
  )
 (:goal (and (passed extra_large_clamp 29 experimenter_hand 2)))) 