;; problem file: task_bimanual_3.pddl

(define (problem bimanual-prob1)
  (:domain bimanual)
(:objects cup pitcher cracker_box bowl gelatin_box mustard_bottle pudding_box tomato_soup_can tuna_fish_can sugar_box initial_location_handleft initial_location_handright - graspable 
            handright handleft - hand
            15 20 - location
  )

  (:init 
  	(at handleft initial_location_handleft)
    (at handright initial_location_handright)
  )

 (:goal (and (poured pitcher 15 cup 20)))) 