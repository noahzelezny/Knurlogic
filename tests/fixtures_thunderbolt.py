"""Real outputs from the M3 Ultra / M4 Max rig (2026-09-27), trimmed to
the fields cluster/links reads. Two cables join them: Thunderbolt 5 at
80 Gb/s (M3 en7 "Thunderbolt 6", receptacle 6, 198.51.100.1 <-> M4 en2
"Thunderbolt 2", receptacle 2, 198.51.100.2) and Thunderbolt 4 at 40 Gb/s
(M3 en4 "Thunderbolt 3" through a ThunderBay dock, 192.0.2.1 <-> M4 en3
"Thunderbolt 3", 192.0.2.2). RDMA works only over the 80 Gb/s one."""

SP_THUNDERBOLT_M3 = {'SPThunderboltDataType': [{'_items': [{'_name': 'MacBook Pro',
                                        'device_name_key': 'Mac16,5',
                                        'services_title': [{'_name': 'service_ip'},
                                                           {'_name': 'unknown_xd_service'}]}],
                            '_name': 'thunderboltusb4_bus_5',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': '80 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x2',
                                                 'receptacle_id_key': '6',
                                                 'receptacle_status_key': 'receptacle_connected'}},
                           {'_name': 'thunderboltusb4_bus_4',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': 'Up '
                                                                      'to '
                                                                      '120 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x100',
                                                 'receptacle_id_key': '5',
                                                 'receptacle_status_key': 'receptacle_no_devices_connected'}},
                           {'_name': 'thunderboltusb4_bus_3',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': 'Up '
                                                                      'to '
                                                                      '120 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x100',
                                                 'receptacle_id_key': '4',
                                                 'receptacle_status_key': 'receptacle_no_devices_connected'}},
                           {'_items': [{'_items': [{'_name': 'MacBook Pro',
                                                    'device_name_key': 'Mac16,5',
                                                    'services_title': [{'_name': 'service_ip'},
                                                                       {'_name': 'unknown_xd_service'}]}],
                                        '_name': 'ThunderBay Flex 8',
                                        'device_name_key': 'ThunderBay '
                                                           'Flex 8',
                                        'mode_key': 'thunderbolt_three',
                                        'receptacle_2_tag': {'current_speed_key': '40 '
                                                                                  'Gb/s',
                                                             'link_status_key': '0x2',
                                                             'receptacle_status_key': 'receptacle_connected'},
                                        'receptacle_upstream_ambiguous_tag': {'current_speed_key': '40 '
                                                                                                   'Gb/s',
                                                                              'link_status_key': '0x2',
                                                                              'receptacle_status_key': 'receptacle_connected'}}],
                            '_name': 'thunderboltusb4_bus_2',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': '40 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x2',
                                                 'receptacle_id_key': '3',
                                                 'receptacle_status_key': 'receptacle_connected'}},
                           {'_items': [{'_name': 'Mercury Helios 3S',
                                        'device_name_key': 'Mercury Helios '
                                                           '3S',
                                        'mode_key': 'thunderbolt_three',
                                        'receptacle_2_tag': {'current_speed_key': 'Up '
                                                                                  'to '
                                                                                  '40 '
                                                                                  'Gb/s',
                                                             'link_status_key': '0x7',
                                                             'receptacle_status_key': 'receptacle_no_devices_connected'},
                                        'receptacle_upstream_ambiguous_tag': {'current_speed_key': '40 '
                                                                                                   'Gb/s',
                                                                              'link_status_key': '0x2',
                                                                              'receptacle_status_key': 'receptacle_connected'}}],
                            '_name': 'thunderboltusb4_bus_1',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': '40 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x2',
                                                 'receptacle_id_key': '2',
                                                 'receptacle_status_key': 'receptacle_connected'}},
                           {'_name': 'thunderboltusb4_bus_0',
                            'device_name_key': 'Mac Studio',
                            'receptacle_1_tag': {'current_speed_key': 'Up '
                                                                      'to '
                                                                      '120 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x100',
                                                 'receptacle_id_key': '1',
                                                 'receptacle_status_key': 'receptacle_no_devices_connected'}}]}

SP_THUNDERBOLT_M4 = {'SPThunderboltDataType': [{'_items': [{'_name': 'Mac Studio',
                                        'device_name_key': 'Mac15,14',
                                        'services_title': [{'_name': 'service_ip'},
                                                           {'_name': 'unknown_xd_service'}]}],
                            '_name': 'thunderboltusb4_bus_2',
                            'device_name_key': 'MacBook Pro',
                            'receptacle_1_tag': {'current_speed_key': '40 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x2',
                                                 'receptacle_id_key': '3',
                                                 'receptacle_status_key': 'receptacle_connected'}},
                           {'_items': [{'_name': 'Mac Studio',
                                        'device_name_key': 'Mac15,14',
                                        'services_title': [{'_name': 'service_ip'},
                                                           {'_name': 'unknown_xd_service'}]}],
                            '_name': 'thunderboltusb4_bus_1',
                            'device_name_key': 'MacBook Pro',
                            'receptacle_1_tag': {'current_speed_key': '80 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x2',
                                                 'receptacle_id_key': '2',
                                                 'receptacle_status_key': 'receptacle_connected'}},
                           {'_name': 'thunderboltusb4_bus_0',
                            'device_name_key': 'MacBook Pro',
                            'receptacle_1_tag': {'current_speed_key': 'Up '
                                                                      'to '
                                                                      '120 '
                                                                      'Gb/s',
                                                 'link_status_key': '0x100',
                                                 'receptacle_id_key': '1',
                                                 'receptacle_status_key': 'receptacle_no_devices_connected'}}]}

PORTS_M3 = """
Hardware Port: Ethernet
Device: en0

Hardware Port: Ethernet Adapter (en8)
Device: en8

Hardware Port: Ethernet Adapter (en9)
Device: en9

Hardware Port: Ethernet Adapter (en10)
Device: en10

Hardware Port: Ethernet Adapter (en11)
Device: en11

Hardware Port: Ethernet Adapter (en12)
Device: en12

Hardware Port: Ethernet Adapter (en13)
Device: en13

Hardware Port: Wi-Fi
Device: en1

Hardware Port: Thunderbolt 1
Device: en2

Hardware Port: Thunderbolt 2
Device: en3

Hardware Port: Thunderbolt 3
Device: en4

Hardware Port: Thunderbolt 4
Device: en5

Hardware Port: Thunderbolt 5
Device: en6

Hardware Port: Thunderbolt 6
Device: en7

VLAN Configurations
===================
"""

PORTS_M4 = """
Hardware Port: Ethernet Adapter (en4)
Device: en4

Hardware Port: USB 10/100/1G/2.5G LAN
Device: en7

Hardware Port: Ethernet Adapter (en5)
Device: en5

Hardware Port: Ethernet Adapter (en6)
Device: en6

Hardware Port: Wi-Fi
Device: en0

Hardware Port: Thunderbolt 1
Device: en1

Hardware Port: Thunderbolt 2
Device: en2

Hardware Port: Thunderbolt 3
Device: en3

VLAN Configurations
===================
"""
