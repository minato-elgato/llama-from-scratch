from manim import *

class RoPEVisualization(Scene):
    def construct(self):
        # 1. Setup the structural grid (Dark Mode Aesthetic)
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#30363d", "stroke_opacity": 0.5}
        )
        self.play(Create(plane), run_time=1.5)

        # 2. Define the initial Query Vector (Before Rotation)
        # Simulating a 2D slice of the head dimension
        initial_vector = Vector([2, 1], color="#00f2fe")
        vector_label = MathTex(r"\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}", color="#00f2fe").next_to(initial_vector.get_end(), RIGHT)
        
        self.play(GrowArrow(initial_vector), Write(vector_label))
        self.wait(1)

        # 3. Introduce the Rotation Mathematics
        title = Tex("Rotary Positional Embedding (RoPE)", color=WHITE, font_size=48).to_edge(UP)
        math_eq = MathTex(
            r"\mathbf{R}_{\Theta, m}^2 \mathbf{x} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \mathbf{x}", 
            color="#fe007a"
        ).next_to(title, DOWN)
        
        self.play(Write(title), Write(math_eq))
        self.wait(1)

        # 4. Animate the Transformation (Encoding position m)
        # Let's rotate it by an arbitrary angle simulating the position multiplier m*theta
        angle = PI / 3 
        rotated_vector = Vector([2 * np.cos(angle) - 1 * np.sin(angle), 2 * np.sin(angle) + 1 * np.cos(angle)], color="#fe007a")
        rotated_label = MathTex(r"\mathbf{x}_{rotated}", color="#fe007a").next_to(rotated_vector.get_end(), UP)

        self.play(
            Transform(initial_vector, rotated_vector),
            Transform(vector_label, rotated_label),
            run_time=2,
            path_arc=angle # Makes the vector sweep in a circular path
        )
        self.wait(2)

        # 5. Fade out clean
        self.play(FadeOut(Group(plane, initial_vector, vector_label, title, math_eq)))