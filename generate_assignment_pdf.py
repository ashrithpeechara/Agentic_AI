import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Top Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Coding Assignment 1 - Applied Agentic AI (LLM & RAG)")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        
        # Bottom Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 30, page_str)
        self.drawString(54, 30, "Applied Agentic AI - Lab & Assignment Submission")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 42, 8.5 * inch - 54, 42)
        
        self.restoreState()

def create_assignment_pdf(output_filename, student_name="Ashrith Peechara", roll_no="2311CS040137"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
        spaceBefore=2
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
    )

    desc_style = ParagraphStyle(
        'QDesc',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1A202C"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    code_style = ParagraphStyle(
        'CodeText',
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # Student Info & Title Header Table
    header_data = [
        [Paragraph(f"<b>Name:</b> {student_name}", meta_style), Paragraph(f"<b>Roll No:</b> {roll_no}", meta_style)],
        [Paragraph("<b>Course:</b> Applied Agentic AI (LLM & RAG)", meta_style), Paragraph("<b>Assignment:</b> Coding Assignment 1", meta_style)]
    ]
    header_table = Table(header_data, colWidths=[260, 252])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    
    story.append(header_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Coding Assignment 1 - Applied Agentic AI (LLM & RAG)", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=8))

    # Read the 4 tasks code from assignments folder
    assignments_dir = r"c:\agentic_ai_assignment\assignments"
    
    with open(os.path.join(assignments_dir, "llm-workflow.py"), "r", encoding="utf-8") as f:
        code_t1 = f.read()

    with open(os.path.join(assignments_dir, "prompt-chaining.py"), "r", encoding="utf-8") as f:
        code_t2 = f.read()

    with open(os.path.join(assignments_dir, "agentic-ai.py"), "r", encoding="utf-8") as f:
        code_t3 = f.read()

    with open(os.path.join(assignments_dir, "rag_ai.py"), "r", encoding="utf-8") as f:
        code_t4 = f.read()

    tasks = [
        (
            "1. LLM Workflow: Develop a Python program that accepts user input and generates a response using an LLM (OpenAI/Gemini/Ollama/Groq).",
            code_t1
        ),
        (
            "2. Prompt Chaining: Implement a multi-step LLM workflow to generate a summary, extract key points, and produce three questions from a given topic.",
            code_t2
        ),
        (
            "3. Agentic AI: Build a simple AI agent that accepts a task, plans the required steps, executes them, and displays the final output.",
            code_t3
        ),
        (
            "4. RAG-Based Question Answering: Develop a basic RAG application that retrieves relevant information from a PDF/TXT document and answers user queries using an LLM.",
            code_t4
        )
    ]

    for title, code_text in tasks:
        story.append(Paragraph(title, desc_style))
        story.append(Spacer(1, 3))
        
        # Preformatted code block
        code_p = Preformatted(code_text.strip(), code_style)
        story.append(code_p)
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_filename}")

if __name__ == "__main__":
    create_assignment_pdf(r"c:\agentic_ai_assignment\assignment_1_agentic_ai.pdf", student_name="Ashrith Peechara", roll_no="2311CS040137")
    create_assignment_pdf(r"c:\agentic_ai_assignment\assignment_1_agentic_ai-2311CS040137.pdf", student_name="Ashrith Peechara", roll_no="2311CS040137")
    print("Done!")

