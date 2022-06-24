import intersect


# medida1 = (119, 529)
# medida2 = (25, 344)

# start_point = (31, 330)
# end_point = (111, 572)

medida1 = (190, 548)
medida2 = (534, 425)

end_point = (355.2663364686633, 467.6676402050421)
start_point = (368.7336635313367, 505.3323597949579)


DoIntersect, orientacao = intersect.doIntersect(
    medida1, medida2, start_point, end_point)

print(DoIntersect, orientacao)
