import pandas as pd

mat = pd.read_csv('student-mat.csv', sep=';')
por = pd.read_csv('student-por.csv', sep=';')

id_cols = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian']
merged = pd.merge(mat, por, on=id_cols, suffixes=('_mat','_por'))

print(f"Estudiantes en MAT: {len(mat)}")
print(f"Estudiantes en POR: {len(por)}")
print(f"Estudiantes en COMUN: {len(merged)}")
print()

g0 = mat[mat['G3'] == 0]
print(f"G3=0 en MAT: {len(g0)} estudiantes")
print(f"  G1 > 0 (abandonaron en el camino): {(g0['G1'] > 0).sum()}")
print(f"  G1 = 0 (no tuvieron calificacion): {(g0['G1'] == 0).sum()}")
print()
print("Detalle G1, G2 cuando G3=0:")
print(g0[['G1','G2','G3','absences','failures']].to_string())
