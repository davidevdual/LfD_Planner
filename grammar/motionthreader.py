import AST
from AST import addToClass
from motionparser import MotionParser
import sys,os

class MotionThreader(object):

    '''
    @addToClass(AST.Node)
    def thread_ast(self,lastNode):
        print("lastNode = ",lastNode,"children : ",self.children);
        for c in self.children:
            lastNode = c.thread_ast(lastNode);
        lastNode.addNext(self);
        return self;
    '''
    # @input tree The Abstract Syntax Tree object
    def thread(self,tree):
        entry = AST.EntryNode();
        ast = tree.thread_ast(entry);
        return entry;

def main():
    motionthreader = MotionThreader();
    motionparser = MotionParser();
    ast,manipulated_object = motionparser.parse("approach handleft bleach_cleanser and enclose handleft bleach_cleanser and approach handright bleach_cleanser to open bleach_cleanser 5");
    #print("ast: ",ast);
    entry = motionthreader.thread(ast);
    #print("entry next nodes = ",entry.next);
    graph = ast.makegraphicaltree();
    entry.threadTree(graph);
    name = os.path.splitext('graph−ast.pdf');
    graph.write_pdf('graph−ast.pdf');

if __name__=="__main__":
    main();