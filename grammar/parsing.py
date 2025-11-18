#################################################################
## This code corresponds to the Action Grammar parser. The 
## grammar rules are defined in this file.
#################################################################
#################################################################
## Author: David Carmona-Moreno
## Copyright: Copyright 2020, Dual-Arm project
## Version: v1.1
## Maintainer: David Carmona-Moreno
## Email: e0348847@u.nus.edu
## Status: First stable release
#################################################################

import ply.yacc as yacc
import ply.lex as lex
import command as cmd

# Get the token map from the lexer.  This is required.
from mylexer import MyLexer

## @class Parsing
#  @brief Parse a sentence upon the Action Grammar
class Parsing(object):
	
	def __init__(self,_is_simulation,_is_vision):
		self.lexer = MyLexer();
		self.tokens = self.lexer.tokens;
		self.parser = yacc.yacc(module=self, debug=True);
		self.is_simulation = _is_simulation;
		self.is_vision = _is_vision;
		self.command = cmd.Command(self.is_simulation,self.is_vision);

	def p_error(self, p):
		if p:
			stack_state_str = " ".join([symbol.type for symbol
	                                    in self.parser.symstack[1:]])
			raise Exception("Syntax error at '%s', type %s, on line %d\n"
				"Parser state: %s %s . %s" %
				(p.value, p.type, p.lineno,
				self.parser.state, stack_state_str, p))
		else:
			raise Exception("Syntax error at '%s', type %s, on line %d\n"
                "Parser state: %s %s . %s" %
                (p.value, p.type, p.lineno,
                 self.parser.state, stack_state_str, p))

   	## A'' rule
	#  @param p      A string 
	def p_actions_actionSecond_empty(self, p):
		'actionSecond : actionFirst'
		p[0] = p[1];

   	## A'' rule
	#  @param p      A string 
	def p_actions_actionSecond(self, p):
		'actionSecond : actionFirst MODIFIERGOAL GOAL'
		p[0] = p[1] + p[2] + p[3];
	
   	## A' rule
	#  @param p      A string 
	def p_actions_objectcomplement(self, p):
		'actionFirst : actionFirst MODIFIER OBJECTCOMPLEMENT actionFirst'
		if p[4] is None:
			p[0] = p[1] + " " + p[2] + " " + p[3];
			self.command.actions_objectcomplement(p[1],p[2],p[3]);
		else:
			p[0] = p[1] + " " + p[2] + " " + p[3] + " " + p[4];

   	## A' rule
	#  @param p      A string 
	def p_actions_actionFirst(self, p):
		'actionFirst : MODIFIER actionFirst'
		p[0] = p[1] + " " + p[2];
	
   	## A' rule
	#  @param p      A string 
	def p_actions_toolcomplement(self, p):
		'actionFirst : ACTION TOOLCOMPLEMENT'
		p[0] = p[1] + " " + p[2];
		self.command.actions_toolcomplement(p[1],p[2]);
		print("In p_actions_toolcomplement:",p[0]);

   	## A' rule
	#  @param p      A string 
	def p_actions_actionFirst_empty(self, p):
		'actionFirst : empty'
		p[0] = p[1];

   	## Empty production
	#  @param p      A string 
	def p_empty(self, p):
		'empty :'
		pass

   	## Parsing the text
	#  @param text      A string 
	def parse(self, text):
		return self.parser.parse(text, self.lexer);

	## Notify the Command class that the parsing has been achieved
	def parse_done(self):
		self.command.parsing_done();# The parsing is done. The commands can be sent to the low-level controller

   	## Access the parser
	def getParser(self):
		return self.parser

	## Execute the parser on a sentence
	#	@param sentence The sentence to parse
	def runParser(self,sentence):
		# Get the parser for the actions grammar
		#parsing = Parsing();
		#parser = parsing.getParser();
		k = 0;

		# Due to the nature of the action grammar, the parsing has to be applied
		## several times to execute all the actions correctly.
		while k != 3:
			try:
				#s = 'approach leftarm to cup and enclose lefthand around cup and approach
				#righthand to pitcher and enclose righthand around pitcher and approach
				#righthand_pitcher to lefthand_cup for pouring'
				s = sentence;  
			except EOFError:
				break;
			if not s: continue
			result = self.parser.parse(s);# Parse the sentence
			k = k + 1;