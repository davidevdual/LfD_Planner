import ply.yacc as yacc
from motionlexer import MotionLexer
import os
import AST

class MotionParser(object):

    def __init__(self):
        self.lexer = MotionLexer();
        self.tokens = self.lexer.tokens;
        self.parser = yacc.yacc(start='plan',module=self, debug=True);
        self.manipulated_object = "";

    def p_plan_spatialPositions_openOnly(self,p):
        '''plan : actionFirst INFINITIVE GOAL OBJECTCOMPLEMENT SPATIALOCATION'''
        p[0] = AST.PlanNode(p[3],[p[1],AST.TokenNode(p[3])]);
        self.manipulated_object = p[4];

    def p_plan_spatialPositions(self,p):
        '''plan : actionFirst INFINITIVE GOAL OBJECTCOMPLEMENT SPATIALOCATION PREPOSITION OBJECTCOMPLEMENT SPATIALOCATION'''
        p[0] = AST.PlanNode(p[3],[p[1],AST.TokenNode(p[3])]);
        self.manipulated_object = p[4];

    def p_actionFirst_conj(self,p):
        '''actionFirst : actionFirst CONJUNCTION actionFirst'''
        p[0] = AST.ConjNode(p[2],[p[1],p[3]]);

    def p_actionFirst_complements(self,p):
        '''actionFirst : ACTION TOOLCOMPLEMENT OBJECTCOMPLEMENT'''
        print("p[0] = ",p[0]);
        p[0] = AST.ActionNode(p[1],[AST.TokenNode(p[2]),AST.TokenNode(p[3])]);

    def p_error(self,p):
        raise Exception("Syntax Error");

    def parse(self,plan):
        return self.parser.parse(plan),self.manipulated_object;

def main():
    motionparser = MotionParser();
    result,manipulated_object = motionparser.parse("approach handright pitcher_base and enclose handright pitcher_base and approach handleft mug and enclose handleft mug and approach handright mug to pour pitcher_base 4 into mug 14");
    graph = result.makegraphicaltree();
    name = os.path.splitext('graph−ast.pdf');
    graph.write_pdf('graph−ast.pdf');
    #print(result);

if __name__=="__main__":
    main();
