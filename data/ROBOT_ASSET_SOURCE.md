# G1 training and MuJoCo assets

These selected public robot files are copied byte-for-byte from the upstream BeyondMimic asset archive: https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz .

Archive SHA-256: b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674 . The source package.xml declares BSD and names Qiayuan Liao as author/maintainer. Original package.xml is retained. The full archive is not included.

configs/g1_assets.json records each selected URDF/MJCF/mesh file and checksum. These are the same G1 bytes already used for the local URDF parity audit. Bundling the required files follows the existing G1 training repository pattern and avoids a live Google Storage download during a charged GM run. No robot, dynamics, rewards or motion reference is changed.
