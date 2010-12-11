<?xml version='1.0' encoding='utf-8'?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:output method='text'/>

<xsl:template match='/'>
  <xsl:apply-templates select='/tagset/lfm[2]/toptags'/>
</xsl:template>

<xsl:template match='toptags'>
  <xsl:message>Transforming annotations for: <xsl:value-of select='@artist'/> -- <xsl:value-of select='@track'/></xsl:message>
  <xsl:apply-templates select='tag'/>
</xsl:template>

<xsl:template match='tag'>
<xsl:value-of select='count'/><xsl:text> </xsl:text><xsl:value-of select='name'/><xsl:text>
</xsl:text>
</xsl:template>

</xsl:stylesheet>

