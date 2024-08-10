import cv2
import matplotlib.pyplot as plt
import numpy as np

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

class Mirror_Symmetry_detection:
    def __init__(self, image_path: str):
        self.image = self._read_color_image(image_path)  
        self.reflected_image = np.fliplr(self.image) 
        
        
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None) 
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)
    
    def _read_color_image(self, image_path):
        
        image = cv2.imread(image_path)  
        b, g, r = cv2.split(image)    
        image = cv2.merge([r, g, b])
        return image
    
    def find_matchpoints(self):
        
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        matchpoints = [item[0] for item in matches]
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)
        return matchpoints
    
    def find_points_r_theta(self, matchpoints:list):
        
        points_r = []
        points_theta = []
        for match in matchpoints:
            point = self.kp1[match.queryIdx]
            mirpoint = self.kp2[match.trainIdx]
            
            mirpoint.angle = np.deg2rad(mirpoint.angle)
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi
                
            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
            
            theta = angle_with_x_axis(point.pt, mirpoint.pt)
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)
            
            points_r.append(r)
            points_theta.append(theta)
            
        return points_r, points_theta

    
    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
       
        for k, v in sorted_vote.items():
            if vertical:
                return k[0], k[1]
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]
    
    def sort_hexbin_by_votes(self, image_hexbin):
        
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts))
        verts = image_hexbin.get_offsets()
        output = {}
        
        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
    
    def draw_mirrorLine(self, r, theta, title:str): 
        
        mirror_line_image = np.copy(self.image)
        
        for y in range(len(self.image)): 
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < mirror_line_image.shape[1]:
                    mirror_line_image[y][x] = [255, 0, 0]  # Red line for better visibility
                    mirror_line_image[y][x + 1] = [255, 0, 0] 
            except IndexError:
                continue
            
        plt.imshow(mirror_line_image)
        plt.axis('off') 
        plt.title(title)
        plt.show()
        
    def check_symmetry(self, points_r):
        
        return len(points_r) > 0  # Adjust this threshold as needed

def angle_with_x_axis(pi, pj):
    
    x, y = pi[0] - pj[0], pi[1] - pj[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

def detecting_mirrorLine(picture_name: str, title: str, show_detail=False):
   
    mirror = Mirror_Symmetry_detection(picture_name)
    matchpoints = mirror.find_matchpoints()
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)

    if not mirror.check_symmetry(points_r):
        plt.figure()
        plt.text(0.5, 0.5, 'No symmetry detected', fontsize=20, ha='center')
        plt.axis('off')
        plt.show()
        return


    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r) 
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)
    mirror.draw_mirrorLine(r, theta, title)

if __name__ == "__main__":
    detecting_mirrorLine('C:/Users/Eashan/Desktop/adobegensolve/Symmetry/tree.png', "Symmetry Mirror Line", show_detail=True)
