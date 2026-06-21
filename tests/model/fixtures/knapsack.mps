* SCIP STATISTICS
*   Problem name     : knapsack
*   Variables        : 5 (5 binary, 0 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 1
NAME          knapsack
OBJSENSE
  MAX
ROWS
 N  Obj 
 L  capacity 
COLUMNS
    INTSTART  'MARKER'                            'INTORG'                           
    item0     Obj                              8  capacity                         5 
    item1     Obj                              5  capacity                         3 
    item2     Obj                             11  capacity                         7 
    item3     Obj                              6  capacity                         4 
    item4     Obj                              9  capacity                         6 
    INTEND    'MARKER'                            'INTEND'                           
RHS
    RHS       capacity                        12 
BOUNDS
 BV Bound     item0                              
 BV Bound     item1                              
 BV Bound     item2                              
 BV Bound     item3                              
 BV Bound     item4                              
ENDATA