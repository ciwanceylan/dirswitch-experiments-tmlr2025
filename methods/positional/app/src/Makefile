JAVAC=javac
.SUFFIXES: .java .class
.java.class:
	$(JAVAC) $*.java
SRCS = \
	PPREmbedding.java \
	SimilarityRanker.java

default: classes

classes: $(SRCS:.java=.class)

clean:
	$(RM) *.class
