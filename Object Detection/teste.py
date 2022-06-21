import intersect


# medida1 = (119, 529)
# medida2 = (25, 344)

# start_point = (31, 330)
# end_point = (111, 572)

medida1 = (551,491)
medida2 = (184, 415)

start_point = (117, 553)
end_point = (534, 425)


DoIntersect, orientacao = intersect.doIntersect(
    medida1, medida2, start_point, end_point)

print(DoIntersect, orientacao)
