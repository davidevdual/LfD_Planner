import ply.lex as lex

class MotionLexer(object):

    # All the tokens' types for the dual-arm language used by the robot
    tokens = (
        'INFINITIVE',
        'GOAL',
        'CONJUNCTION',
        'ACTION',
        'TOOLCOMPLEMENT',
        'OBJECTCOMPLEMENT',
        'PREPOSITION',
        'SPATIALOCATION'
    );

    # Regular expressions to extract the token values from the language
    t_INFINITIVE = r'\w+(?=\s*(pour|pass|open)[^/])';
    t_PREPOSITION = r'\b(into)\b|(\w+(?=\s*(experimenter_hand)[^/]))';
    t_SPATIALOCATION = r'\b\d+\b';
    t_GOAL = r'\b(pour|pass|open)\b';
    t_CONJUNCTION = r'\b(and)\b';
    t_ACTION = r'\b(approach|enclose|raise)\b';
    t_TOOLCOMPLEMENT = r'\b(handleft|handright)\b';
    t_OBJECTCOMPLEMENT = r'\b(mug|bottle|pitcher_base|bowl|cracker_box|experimenter_hand|master_chef_can|mustard_bottle|bleach_cleanser|tomato_soup_can)\b';
    t_ignore = ' \t';# Ignore spaces

    ## Detect a new line
	#  @param t      The string on which to perform the lexical analysis
    def t_newline(self,t) :
        r'\n+';
        t.lexer.lineno += len(t.value);

    ## Overwrite the error function to throw an error in case of bad lexical analysis
	#  @param t      The string on which to perform the lexical analysis
    def t_error(self,t) :
        print("Illegal character '% s'" % t. value[0]);
        t.lexer.skip(1);

    def build(self,	**kwargs):
        self.lexer = lex.lex(module=self, **kwargs);

    def test(self, data):
        self.lexer.input(data)
        while True:
            tok = self.lexer.token();
            if not tok: break
            print("lines %d: %s(%s)" % (tok.lineno,tok.type,tok.value));

    def get_tokens(self):
        return self.tokens;

    def __init__(self):
        self.lexer = lex.lex(module=self);

def main():
    motionlexer = MotionLexer();
    motionlexer.test("approach handright master_chef_can to pass master_chef_can 13 to experimenter_hand 6");

    while 1:
        tok = lex.token()
        if not tok: break
        print("lines %d: %s(%s)" % (tok.lineno,tok.type,tok.value));

if __name__=="__main__":
    main();