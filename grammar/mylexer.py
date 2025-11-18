#################################################################
## This code contains the lexer which is a class holding all 
## the terminals/tokens of the Action Grammar.
#################################################################
#################################################################
## Author: David Carmona-Moreno
## Copyright: Copyright 2020, Dual-Arm project
## Version: v1.1
## Maintainer: David Carmona-Moreno
## Email: e0348847@u.nus.edu
## Status: First stable release
#################################################################

import ply.lex as lex

## @class MyLexer
#  @brief Lexer containing all the terminals/tokens of the Action Grammar
class MyLexer(object):

	# List of token names.   This is always required
	tokens = (
	   "GOAL",
	   "ACTION",
	   "MODIFIER",
	   "MODIFIERGOAL",
	   "OBJECTCOMPLEMENT",
	   "TOOLCOMPLEMENT",
	   "SENSOREVENT",
	)

	## Method defining the goals
	#  @param t      	   A string
	#  @return             A string
	def t_GOAL(self, t):
		r'\b(opening|passing|pouring|screwing)\b'
		return t

	## Method defining the primitive actions
	#  @param t            A string
	#  @return             A string
	def t_ACTION(self, t):
		r'\b(approach|enclose)\b'
		return t

	## Method defining the modifiers
	#  @param t            A string
	#  @return             A string
	def t_MODIFIER(self, t):
		r'\b(with|on|to|and|at_the_same_time|the|around)\b'
		return t

	## Method defining the goal modifier
	#  @param t            A string
	#  @return             A string
	def t_MODIFIERGOAL(self, t):
		r'\b(for)\b'
		return t

	## Method defining the object complements
	#  @param t            A string
	#  @return             A string
	def t_OBJECTCOMPLEMENT(self, t):
		r'\b(banana|cup|pitcher|lefthand_cup)\b'
		return t

	## Method defining the tool complements
	#  @param t            A string
	#  @return             A string
	def t_TOOLCOMPLEMENT(self, t):
		r'\b(lefthand|righthand|righthand_pitcher|lefthand_cup)\b'
		return t

	## Method to skip a new line
	#  @param t            A string
	#  @return             A string
	def t_newline(self, t):
		r'\n+'
		t.lexer.lineno += len(t.value)

	# A string containing ignored characters (spaces and tabs)
	t_ignore  = ' \t'

	## Method handling with the grammatical errors
	#  @param t            A string
	#  @return             A string
	def t_error(self, t):
		print("Illegal character '%s'" % t.value[0])
		t.lexer.skip(1)

	def build(self,	**kwargs):
		self.lexer = lex.lex(module=self, **kwargs)
		
	def test(self, data):
		self.lexer.input(data)
		while True:
			tok = self.lexer.token()
			if not tok: 
				break
				print(tok)

	def __init__(self):
		self.lexer = lex.lex(module=self)