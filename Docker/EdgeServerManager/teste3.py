def main():
    orig = [
        {
            "name": "interior salas",
            "start_point": [
                    177,
                    553
            ],
            "end_point": [
                534,
                425
            ],
            "name_zone_before": "outside",
            "name_zone_after": "inside",
            "id_association": {}
        },
        {
            "name": "123",
            "start_point": [
                    31,
                    330
            ],
            "end_point": [
                111,
                572
            ],
            "name_zone_before": "antes",
            "name_zone_after": "depois",
            "id_association": {
                "publish_location": "edge_config/trackingcamera2",
                "name": "Escola de Engenharia"
            }
        }
    ]
    

    extra = [{'name': 'interior salas', 'start_point': [177, 553], 'end_point': [534, 425], 'name_zone_before': 'outside', 'name_zone_after': 'inside'}, {
        'name': '123', 'start_point': [31, 330], 'end_point': [111, 572], 'name_zone_before': 'antes', 'name_zone_after': 'depois'},{}]
    
    for index, value in enumerate(extra):
        if len(orig)<=index:
            orig.append(value)
        orig[index].update(value)
    
    print(orig)

if __name__ == '__main__':
    main()
