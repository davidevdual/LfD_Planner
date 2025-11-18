;; domain_3.pddl for bimanual manipulation
;; This domain is for the passing goal exclusively

(define (domain domain_3)
  (:requirements :strips)

  (:types graspable - object
          hand - object
          experimenterHand - hand
          location - object
  )

  (:predicates 
  	(at ?hand - hand ?obj - graspable)
    (approached ?hand - hand ?obj - graspable)
    (enclosed ?hand - hand ?obj - graspable)
    (passed ?obj - graspable ?objLoc - location ?expHand - hand ?expHandLoc - location)
  )

  (:action approach
           :parameters (?hand - hand ?obj - graspable)
           :precondition (and (not (at ?hand ?obj)))
           :effect (and (at ?hand ?obj))
   )

   (:action enclose
           :parameters (?hand - hand ?obj - graspable)
           :precondition (and (at ?hand ?obj) (not (enclosed ?hand ?obj)))
           :effect (and (enclosed ?hand ?obj))
   )

   (:action pass
           :parameters (?hand1 ?hand2 - hand ?obj - graspable ?objLoc - location ?expHand - experimenterHand ?expHandLoc - location)
           :precondition (and (enclosed ?hand1 ?obj) (enclosed ?hand2 ?obj) (not (= ?hand1 ?hand2)) (not (= ?hand1 ?expHand)) (not (= ?hand2 ?expHand)) (not (at ?expHand ?obj)) (not (at ?expHand ?objLoc)))
           :effect (and (passed ?obj ?objLoc ?expHand ?expHandLoc))
   )
)