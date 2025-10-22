# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

"""
Legacy wrapper for optical flow consistency checking.

This module now imports from the refactored deforum/utils/optical_flow_utils.py
for better maintainability and testability.

Original implementation taken from https://github.com/Sxela/flow_tools/blob/main
(GNU GPL Licensed), and modified to suit Deforum.
"""

# Import pure function from refactored utils module
from deforum.utils.optical_flow_utils import make_consistency


# parser = argparse.ArgumentParser()
# parser.add_argument("--flow_fwd", type=str, required=True, help="Forward flow path or glob pattern")
# parser.add_argument("--flow_bwd", type=str, required=True, help="Backward flow path or glob pattern")
# parser.add_argument("--output", type=str, required=True, help="Output consistency map path")
# parser.add_argument("--output_postfix", type=str, default='_cc', help="Output consistency map name postfix")
# parser.add_argument("--image_output", action='store_true', help="Output consistency map as b\w image path")
# parser.add_argument("--skip_numpy_output", action='store_true', help="Don`t save numpy array")
# parser.add_argument("--blur", type=float, default=2., help="Gaussian blur kernel size (0 for no blur)")
# parser.add_argument("--bottom_clamp", type=float, default=0., help="Clamp lower values")
# parser.add_argument("--edges_reliable", action='store_true', help="Consider edges reliable")
# parser.add_argument("--save_separate_channels", action='store_true', help="Save consistency mask layers as separate channels")
# args = parser.parse_args()

# def run(args):
#   flow_fwd_many = sorted(glob.glob(args.flow_fwd))
#   flow_bwd_many = sorted(glob.glob(args.flow_bwd))
#   if len(flow_fwd_many)!= len(flow_bwd_many): 
#     raise Exception('Forward and backward flow file numbers don`t match')
#     return
  
#   for flow_fwd,flow_bwd in tqdm(zip(flow_fwd_many, flow_bwd_many)):
#     flow_fwd = flow_fwd.replace('\\','/')
#     flow_bwd = flow_bwd.replace('\\','/')
#     flow1 = np.load(flow_fwd)
#     flow2 = np.load(flow_bwd)
#     consistency_map_multilayer = make_consistency(flow1, flow2, edges_unreliable=not args.edges_reliable)
    
#     if args.save_separate_channels:  
#           consistency_map = consistency_map_multilayer
#     else:
#           consistency_map = np.ones_like(consistency_map_multilayer[...,0])
#           consistency_map*=consistency_map_multilayer[...,0]
#           consistency_map*=consistency_map_multilayer[...,1]
#           consistency_map*=consistency_map_multilayer[...,2] 
          
#     # blur
#     if args.blur>0.:
#       consistency_map = scipy.ndimage.gaussian_filter(consistency_map, [args.blur, args.blur])

#     #clip values between bottom_clamp and 1
#     bottom_clamp = min(max(args.bottom_clamp,0.), 0.999)
#     consistency_map = consistency_map.clip(bottom_clamp, 1)
#     out_fname = args.output+'/'+flow_fwd.split('/')[-1][:-4]+args.output_postfix
      
#     if not args.skip_numpy_output:
#       np.save(out_fname, consistency_map)

#     #save as jpeg 
#     if args.image_output:
#       PIL.Image.fromarray((consistency_map*255.).astype('uint8')).save(out_fname+'.jpg', quality=90)

# run(args)
