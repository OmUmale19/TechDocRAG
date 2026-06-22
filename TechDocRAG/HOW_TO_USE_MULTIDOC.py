"""
Quick Demo: User-Input Multi-Document Q&A
This shows how the new feature works
"""

print("=" * 80)
print("NEW FEATURE: User-Input Multi-Document Q&A")
print("=" * 80)
print()
print("🎯 What's New:")
print()
print("1. When you select 'Multi-Document Q&A', you're asked:")
print("   📝 How many documents? (Max: 4)")
print()
print("2. For EACH document, you provide:")
print("   📄 Document Title")
print("   📝 Document Content (paste or type, press Enter twice to finish)")
print()
print("3. Then ask questions across all your documents!")
print()
print("=" * 80)
print("EXAMPLE WORKFLOW:")
print("=" * 80)
print()
print("Step 1: Select option 3 from main menu")
print()
print("Step 2: Enter number of documents (e.g., '2')")
print()
print("Step 3: For each document:")
print("  → Enter title: 'Invoice 1'")
print("  → Paste content: ")
print("     Invoice No: 001")
print("     Amount: Rs 50,000")
print("     [press Enter twice]")
print()
print("  → Enter title: 'Invoice 2'")
print("  → Paste content:")
print("     Invoice No: 002")
print("     Amount: Rs 75,000")
print("     [press Enter twice]")
print()
print("Step 4: Ask questions:")
print("  • What is the total amount in Invoice 1?")
print("  • Which invoice has the higher amount?")
print("  • Find all invoice numbers")
print()
print("=" * 80)
print("SAMPLE DOCUMENTS YOU CAN USE:")
print("=" * 80)
print()

# Sample 1
print("SAMPLE 1 - Invoice:")
print("-" * 40)
print("""INVOICE NO: INV-2025-001
Date: 10 November 2025
Customer: ABC Corp
Amount: Rs 50,000
Tax: Rs 9,000
Total: Rs 59,000
""")

# Sample 2
print("SAMPLE 2 - Receipt:")
print("-" * 40)
print("""RECEIPT NO: REC-2025-001
Date: 11 November 2025
Customer: XYZ Ltd
Amount Received: Rs 45,000
Payment Mode: UPI
""")

# Sample 3
print("SAMPLE 3 - Order:")
print("-" * 40)
print("""ORDER NO: ORD-2025-001
Date: 12 November 2025
Customer: Tech Solutions
Items: Laptop (2), Mouse (5)
Total Value: Rs 95,000
""")

print()
print("=" * 80)
print("💡 TIPS:")
print("=" * 80)
print()
print("✓ You can paste entire documents at once")
print("✓ Or type line by line")
print("✓ Press Enter twice when done with each document")
print("✓ Ask specific questions about your documents")
print("✓ System searches across ALL documents simultaneously")
print()
print("=" * 80)
print("TO START: Run 'python main.py' and select option 3")
print("=" * 80)
