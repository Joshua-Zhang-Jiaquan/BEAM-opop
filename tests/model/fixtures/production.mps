* SCIP STATISTICS
*   Problem name     : production
*   Variables        : 4 (1 binary, 2 integer, 0 implicit integer, 1 continuous)
*   Constraints      : 3
NAME          production
OBJSENSE
  MIN
ROWS
 N  Obj 
 G  demand 
 L  capacity 
 E  balance 
COLUMNS
    INTSTART  'MARKER'                            'INTORG'                           
    make_a    balance                          1  Obj                              2 
    make_a    demand                           1  capacity                         1 
    make_b    Obj                              3  capacity                         1 
    make_b    demand                           1  balance                         -2 
    line      capacity                      -100  Obj                             10 
    INTEND    'MARKER'                            'INTEND'                           
    buy       Obj                              5  demand                           1 
    buy       balance                          1 
RHS
    RHS       demand                           8  capacity                         6 
    RHS       balance                          3  Obj                             -7 
BOUNDS
 BV Bound     line                               
 UP Bound     make_b                          10 
 UP Bound     make_a                          10 
 UP Bound     buy                             20 
ENDATA